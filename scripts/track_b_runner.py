#!/usr/bin/env python3
"""Track B Run Matrix Runner (Reference Hardware)

Executes XLOG neural examples with strict metric-schema gates and
full 3-seed coverage for all examples.  Records the actual GPU identity
in every artifact so results are always traceable to hardware.

Development hardware: NVIDIA RTX PRO 3000 Blackwell (12 GB, SM120).

Policy:
  - All examples: 3 seeds (7, 42, 123), per-example timeouts
  - Metric JSON must include the frozen schema fields
  - GPU identity is captured and embedded in every metrics.json

Spec: https://xlog.md/guides/benchmarking  (baseline metrics)
      docs-internal/plans/2026-02-16-track-a-run-matrix.md  (run matrix, local-only)
"""

import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORKTREE = Path("/home/dev/projects/xlog/.worktrees/v0.4.0-alpha-integrated")
PYTHON = "/home/dev/projects/xlog/.venv/bin/python"

# Development hardware (for labeling; no hard gate — record whatever GPU is present)
DEV_GPU = "NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU"

DEFAULT_SEEDS = [7, 42, 123]
DEFAULT_TIMEOUT = 1800

# Frozen metric schema — every per-run metrics.json must contain these keys.
FROZEN_SCHEMA_KEYS = {
    "compile_api_sec",
    "first_epoch_sec",
    "steady_epoch_sec_mean",
    "warmup_sec",
    "per_query_ms",
    "epoch_sec",
    "total_train_sec",
}

EXAMPLES = [
    {
        "name": "01_minimal",
        "command": (
            f"{PYTHON} examples/neural/01_minimal/train.py"
            " --engine xlog --epochs 5 --batch-size 64 --seed {seed}"
            " --train-limit 512"
            " --data-path examples/neural/01_minimal/data/mnist"
            " --save-path {run_dir}/mnist_net.pt"
            " --metrics-path {run_dir}/metrics.json"
        ),
        "metric_key": "heldout_addition_acc",
        "manifest": "examples/neural/01_minimal/dataset.json",
        "timeout": 3600,
    },
    {
        "name": "02_coins",
        "command": (
            f"{PYTHON} examples/neural/02_coins/train.py"
            " --mode dev --epochs 12 --batch-size 32 --lr 1e-3 --seed {seed}"
            " --metrics-path {run_dir}/metrics.json"
        ),
        "metric_key": "test_acc",
        "manifest": "examples/neural/02_coins/dataset.json",
    },
    {
        "name": "03_mnist_multidigit",
        "command": (
            f"{PYTHON} examples/neural/03_mnist_multidigit/train.py"
            " --mode dev --epochs 12 --batch-size 32 --lr 1e-3 --seed {seed}"
            " --eval-ratio 0.2"
            " --metrics-path {run_dir}/metrics.json"
        ),
        "metric_key": "eval_joint_proxy",
        "manifest": "examples/neural/03_mnist_multidigit/dataset.json",
    },
    {
        "name": "04_hwf",
        "command": (
            f"{PYTHON} examples/neural/04_hwf/train.py"
            " --mode dev --epochs 12 --batch-size 8 --lr 1e-3 --seed {seed}"
            " --eval-ratio 0.2"
            " --metrics-path {run_dir}/metrics.json"
        ),
        "metric_key": "eval_acc",
        "manifest": "examples/neural/04_hwf/dataset.json",
    },
    {
        "name": "05_poker",
        "command": (
            f"{PYTHON} examples/neural/05_poker/train.py"
            " --mode dev --epochs 20 --batch-size 16 --lr 1e-3 --seed {seed}"
            " --eval-ratio 0.1 --rank-query-weight 1"
            " --metrics-path {run_dir}/metrics.json"
        ),
        "metric_key": "eval_joint_proxy",
        "manifest": "examples/neural/05_poker/dataset.json",
    },
    {
        "name": "06_clutrr",
        "command": (
            f"{PYTHON} examples/neural/06_clutrr/train.py"
            " --mode dev --epochs 10 --batch-size 16 --lr 1e-3 --seed {seed}"
            " --eval-ratio 0.2"
            " --metrics-path {run_dir}/metrics.json"
        ),
        "metric_key": "eval_acc",
        "manifest": "examples/neural/06_clutrr/dataset.json",
    },
]


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def detect_hardware():
    """Detect and record GPU hardware. Fails only if no CUDA GPU is found."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            sys.exit(f"HARDWARE DETECTION FAILED: nvidia-smi returned {result.returncode}")
        gpu_line = result.stdout.strip()
    except FileNotFoundError:
        sys.exit("HARDWARE DETECTION FAILED: nvidia-smi not found")
    except subprocess.TimeoutExpired:
        sys.exit("HARDWARE DETECTION FAILED: nvidia-smi timed out")

    parts = gpu_line.split(", ")
    gpu_name = parts[0] if len(parts) > 0 else "unknown"
    driver = parts[1] if len(parts) > 1 else "unknown"
    vram = parts[2] if len(parts) > 2 else "unknown"
    print(f"Hardware detected: {gpu_name} (driver {driver}, VRAM {vram})", flush=True)
    return gpu_name, driver


def check_schema_gate(metrics_dict, example_name, seed):
    """Validate that a metrics dict contains all frozen schema keys.

    Returns list of missing keys (empty = pass).
    """
    missing = FROZEN_SCHEMA_KEYS - set(metrics_dict.keys())
    if missing:
        print(
            f"  SCHEMA GATE WARNING: {example_name}/seed_{seed} missing keys: "
            f"{sorted(missing)}",
            flush=True,
        )
    return sorted(missing)


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------

def get_env_info(gpu_name, driver):
    """Capture environment metadata."""
    import torch

    git_commit = "unknown"
    git_branch = "unknown"
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(WORKTREE), timeout=5,
        ).stdout.strip()
        git_branch = subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True,
            cwd=str(WORKTREE), timeout=5,
        ).stdout.strip()
    except Exception:
        pass

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch.__version__,
        "pyxlog": "0.4.0",
        "gpu_name": gpu_name,
        "driver": driver,
        "cuda_visible": torch.cuda.is_available(),
    }, {
        "branch": git_branch,
        "commit": git_commit,
    }


def sha256_file(path):
    """Compute SHA-256 of a file."""
    if not os.path.exists(path):
        return "file_not_found"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_final_metric(stdout_path):
    """Parse FINAL_METRIC line from stdout log file."""
    pattern = r"FINAL_METRIC:\s*(\S+)=([\d.]+),\s*threshold=(\S+)"
    if not stdout_path.exists():
        return None, None, None
    text = stdout_path.read_text()
    for line in reversed(text.splitlines()):
        m = re.search(pattern, line)
        if m:
            name = m.group(1)
            value = float(m.group(2))
            thresh_str = m.group(3)
            threshold = None if thresh_str == "none" else float(thresh_str)
            return name, value, threshold
    return None, None, None


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def run_single(example, seed, run_dir, env_info, git_info, run_id):
    """Execute a single training run with file-based I/O."""
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = example["command"].format(seed=seed, run_dir=str(run_dir))
    timeout = example.get("timeout", DEFAULT_TIMEOUT)

    print(f"\n{'='*60}", flush=True)
    print(f"  {example['name']} seed={seed} (timeout={timeout}s)", flush=True)
    print(f"  cmd: {cmd}", flush=True)
    print(f"{'='*60}", flush=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:/usr/local/cuda/lib64"
    env["PYTHONPATH"] = str(WORKTREE)
    env["PYTHONUNBUFFERED"] = "1"

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    start = time.monotonic()
    timed_out = False
    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=fout, stderr=ferr,
            cwd=str(WORKTREE), env=env,
            preexec_fn=os.setsid,
        )
        try:
            exit_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
            exit_code = -1
            timed_out = True
    elapsed = time.monotonic() - start

    if timed_out:
        with open(stderr_path, "a") as f:
            f.write(f"\nTIMEOUT after {timeout}s\n")

    (run_dir / "time.txt").write_text(f"ELAPSED_SEC={elapsed:.2f}\n")
    (run_dir / "exit_code.txt").write_text(str(exit_code))

    # Parse metric from stdout
    metric_name, metric_value, threshold = parse_final_metric(stdout_path)
    if metric_name is None:
        status = "timeout" if timed_out else "missing_metric"
        metric_name = example["metric_key"]
        metric_value = None
        threshold = None
    else:
        status = "ok" if exit_code == 0 else "error"

    manifest_path = str(WORKTREE / example["manifest"])
    manifest_sha = sha256_file(manifest_path)

    gate_pass = None
    if metric_value is not None:
        gate_pass = True if threshold is None else (metric_value >= threshold)

    metrics = {
        "track": "B",
        "run_id": run_id,
        "example": example["name"],
        "seed": seed,
        "status": status,
        "exit_code": exit_code,
        "duration_sec": round(elapsed, 2),
        "command": cmd,
        "metric": {
            "name": metric_name,
            "value": metric_value,
            "threshold": threshold,
        },
        "dataset": {
            "manifest_path": example["manifest"],
            "manifest_sha256": manifest_sha,
            "completeness": "provisional",
        },
        "environment": env_info,
        "git": git_info,
    }

    # Schema gate: training script must write metrics.json with all frozen keys
    ext_metrics_path = run_dir / "metrics.json"
    if not ext_metrics_path.exists():
        schema_missing = sorted(FROZEN_SCHEMA_KEYS)
        print(
            f"  SCHEMA GATE FAIL: {example['name']}/seed_{seed} — "
            f"no metrics.json written by training script",
            flush=True,
        )
    else:
        try:
            ext = json.loads(ext_metrics_path.read_text())
            schema_missing = check_schema_gate(ext, example["name"], seed)
            # Merge frozen-schema fields into our metrics
            for key in FROZEN_SCHEMA_KEYS:
                if key in ext:
                    metrics[key] = ext[key]
            # Preserve optional diagnostic fields
            for opt_key in ("warmup_breakdown",):
                if opt_key in ext:
                    metrics[opt_key] = ext[opt_key]
        except json.JSONDecodeError:
            schema_missing = sorted(FROZEN_SCHEMA_KEYS)
            print(
                f"  SCHEMA GATE FAIL: {example['name']}/seed_{seed} — "
                f"metrics.json is not valid JSON",
                flush=True,
            )

    metrics["schema_gate"] = {
        "passed": len(schema_missing) == 0,
        "missing_keys": schema_missing,
        "frozen_keys": sorted(FROZEN_SCHEMA_KEYS),
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    val_str = f"{metric_value:.4f}" if metric_value is not None else "N/A"
    schema_str = "PASS" if len(schema_missing) == 0 else f"FAIL({schema_missing})"
    print(
        f"  -> {status} | exit={exit_code} | {elapsed:.1f}s | "
        f"{metric_name}={val_str} | schema={schema_str}",
        flush=True,
    )

    return {
        "example": example["name"],
        "seed": seed,
        "status": status,
        "exit_code": exit_code,
        "duration_sec": round(elapsed, 2),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_threshold": threshold,
        "gate_pass": gate_pass,
        "schema_gate_pass": len(schema_missing) == 0,
    }


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary(rows, run_id, out_dir, env_info):
    """Generate summary.csv and summary.json."""
    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "run_id", "example", "seed", "status", "exit_code",
        "duration_sec", "metric_name", "metric_value", "metric_threshold",
        "gate_pass", "schema_gate_pass",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "run_id": run_id})

    examples_agg = {}
    for ex in EXAMPLES:
        name = ex["name"]
        vals = [
            r["metric_value"] for r in rows
            if r["example"] == name and r["metric_value"] is not None
        ]
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            examples_agg[name] = {
                "metric_name": ex["metric_key"],
                "n": len(vals),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }
        else:
            examples_agg[name] = {
                "metric_name": ex["metric_key"],
                "n": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }

    schema_all_pass = all(r["schema_gate_pass"] for r in rows)

    summary = {
        "track": "B",
        "run_id": run_id,
        "hardware": env_info.get("gpu_name", "unknown") if hasattr(env_info, "get") else "unknown",
        "seed_policy": {
            "default": DEFAULT_SEEDS,
        },
        "examples": examples_agg,
        "gates": {
            "schema_gate_all_pass": schema_all_pass,
        },
        "handoff_flags": {
            "scallop_comparison_complete": False,
            "dataset_finalized": False,
        },
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_comparisons(summary, out_dir):
    """Generate comparison artifacts."""
    comp_dir = out_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    xlog_minimal = summary["examples"].get("01_minimal", {})
    if xlog_minimal.get("mean") is not None:
        mnist_comp = {
            "status": "complete",
            "xlog_track_b": {
                "metric_name": xlog_minimal["metric_name"],
                "mean": xlog_minimal["mean"],
                "std": xlog_minimal["std"],
                "n": xlog_minimal["n"],
            },
            "deepproblog_baseline_report": "docs-internal/reports/2026-02-10-deepproblog-baseline-gpu-sequential.md",
        }
    else:
        mnist_comp = {
            "status": "blocked",
            "reason": "01_minimal XLOG run failed; no XLOG metric available",
            "deepproblog_baseline_report": "docs-internal/reports/2026-02-10-deepproblog-baseline-gpu-sequential.md",
        }
    with open(comp_dir / "mnist_vs_deepproblog.json", "w") as f:
        json.dump(mnist_comp, f, indent=2)

    scallop = {
        "available": False,
        "reason": "scallopy/scallop not installed in environment",
        "deferred_to": "post-merge",
    }
    with open(comp_dir / "scallop_status.json", "w") as f:
        json.dump(scallop, f, indent=2)


def generate_manifest(run_id, env_info, git_info, out_dir, total_runs):
    """Generate run_manifest.json."""
    manifest = {
        "track": "B",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed_policy": {
            "default": DEFAULT_SEEDS,
        },
        "examples": [e["name"] for e in EXAMPLES],
        "total_runs": total_runs,
        "environment": env_info,
        "git": git_info,
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("  TRACK B — Reference Hardware Runner", flush=True)
    print("=" * 60, flush=True)

    # ── Detect hardware ──
    gpu_name, driver = detect_hardware()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_track_b"
    out_dir = WORKTREE / "examples" / "neural" / "results" / "track_b" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # All examples get 3 seeds in Track B (no reduced-seed exceptions)
    run_list = []
    for example in EXAMPLES:
        for seed in DEFAULT_SEEDS:
            run_list.append((example, seed))

    print(f"\nRun ID: {run_id}", flush=True)
    print(f"Output: {out_dir}", flush=True)
    print(f"Total runs: {len(run_list)}", flush=True)
    for ex in EXAMPLES:
        timeout = ex.get("timeout", DEFAULT_TIMEOUT)
        print(f"  {ex['name']}: seeds={DEFAULT_SEEDS}, timeout={timeout}s", flush=True)

    env_info, git_info = get_env_info(gpu_name, driver)
    generate_manifest(run_id, env_info, git_info, out_dir, len(run_list))

    rows = []
    for example, seed in run_list:
        run_dir = out_dir / example["name"] / f"seed_{seed}"
        row = run_single(example, seed, run_dir, env_info, git_info, run_id)
        rows.append(row)

    print(f"\n{'='*60}", flush=True)
    print("Generating aggregate artifacts...", flush=True)
    summary = generate_summary(rows, run_id, out_dir, env_info)
    generate_comparisons(summary, out_dir)

    # ── Final report ──
    print(f"\n{'='*60}", flush=True)
    print("TRACK B RESULTS (Reference Hardware)", flush=True)
    print(f"{'='*60}", flush=True)
    total = len(rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    err = sum(1 for r in rows if r["status"] == "error")
    missing = sum(1 for r in rows if r["status"] == "missing_metric")
    tout = sum(1 for r in rows if r["status"] == "timeout")
    schema_pass = sum(1 for r in rows if r["schema_gate_pass"])
    print(
        f"Total: {total} | OK: {ok} | Error: {err} | Timeout: {tout} | "
        f"Missing metric: {missing}",
        flush=True,
    )
    print(f"Schema gate: {schema_pass}/{total} passed", flush=True)
    print(flush=True)
    for name, agg in summary["examples"].items():
        if agg["mean"] is not None:
            print(
                f"  {name}: {agg['metric_name']}={agg['mean']:.4f} "
                f"+/- {agg['std']:.4f} (n={agg['n']})",
                flush=True,
            )
        else:
            print(f"  {name}: NO DATA", flush=True)

    schema_ok = summary["gates"]["schema_gate_all_pass"]
    print(f"\nGates:", flush=True)
    print(f"  Schema:   {'PASS' if schema_ok else 'FAIL'}", flush=True)
    print(f"\nArtifacts: {out_dir}", flush=True)
    print(f"Run ID: {run_id}", flush=True)

    if not schema_ok:
        print(
            "\nFATAL: Some runs failed the schema gate. Training scripts "
            "must emit all frozen schema fields. See per-run metrics.json.",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
