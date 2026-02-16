# Track A Run Matrix and Artifact Schema

> **Purpose:** Execution spec for Track A (functional validation + provisional accuracy metrics now, before charter-compliant Track B reruns).

## Scope and Constraints

- Track A executes on the current machine (`NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU`, driver `573.71`), not RTX 3090.
- Scallop is not currently installed (`scallopy`/`scallop` missing), so Scallop comparison is deferred to Track B.
- DeepProbLog baseline already exists in:
  - `docs/reports/2026-02-10-deepproblog-baseline-gpu-sequential.md`
  - `examples/neural/baseline/results/deepproblog_gpu_sequential/`
- Data exists for all 6 XLOG examples in this worktree, but `02_coins`, `03_mnist_multidigit`, and `04_hwf` look like partial/provisional subsets and must be treated as non-final for milestone claims.

## Track A Objective

- Run all 6 XLOG neural examples end-to-end with held-out evaluation.
- Capture reproducible metrics, timing, and environment metadata in machine-readable artifacts.
- Produce immediate comparison-ready outputs (XLOG now, DeepProbLog reference attached, Scallop deferred).

## Preflight (Required)

Run from:

```bash
cd /home/dev/projects/xlog/.worktrees/v0.4.0-alpha-integrated
```

Environment:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64
export PYTHON=/home/dev/projects/xlog/.venv/bin/python
```

Sanity checks:

```bash
$PYTHON -c "import pyxlog, torch; print('pyxlog ok:', pyxlog.__file__); print('cuda:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
git rev-parse HEAD
```

## Run Matrix (Track A)

Profile:

- Mode: `dev` for `02..06` (full-data path, no release gate abort).
- Seeds: `7`, `42`, `123` (default). 01_minimal: seed `42` only (per-query host sync overhead in forward_backward).
- Each run writes its own logs and JSON metrics.

### Matrix

| Example | Script | Command Template | Final Metric Key |
|---|---|---|---|
| `01_minimal` | `examples/neural/01_minimal/train.py` | `$PYTHON examples/neural/01_minimal/train.py --engine xlog --epochs 5 --batch-size 64 --seed {seed} --train-limit 512 --data-path examples/neural/01_minimal/data/mnist --save-path {run_dir}/mnist_net.pt` | `heldout_addition_acc` |
| `02_coins` | `examples/neural/02_coins/train.py` | `$PYTHON examples/neural/02_coins/train.py --mode dev --epochs 12 --batch-size 32 --lr 1e-3 --seed {seed}` | `test_acc` |
| `03_mnist_multidigit` | `examples/neural/03_mnist_multidigit/train.py` | `$PYTHON examples/neural/03_mnist_multidigit/train.py --mode dev --epochs 12 --batch-size 32 --lr 1e-3 --seed {seed} --eval-ratio 0.2` | `eval_joint_proxy` |
| `04_hwf` | `examples/neural/04_hwf/train.py` | `$PYTHON examples/neural/04_hwf/train.py --mode dev --epochs 12 --batch-size 8 --lr 1e-3 --seed {seed} --eval-ratio 0.2` | `eval_acc` |
| `05_poker` | `examples/neural/05_poker/train.py` | `$PYTHON examples/neural/05_poker/train.py --mode dev --epochs 20 --batch-size 16 --lr 1e-3 --seed {seed} --eval-ratio 0.1 --rank-query-weight 1` | `eval_joint_proxy` |
| `06_clutrr` | `examples/neural/06_clutrr/train.py` | `$PYTHON examples/neural/06_clutrr/train.py --mode dev --epochs 10 --batch-size 16 --lr 1e-3 --seed {seed} --eval-ratio 0.2` | `eval_acc` |

## Execution Procedure

Use one run id for all results:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_track_a_dev"
OUT="examples/neural/results/track_a/${RUN_ID}"
mkdir -p "$OUT"
```

For each `{example, seed}`:

1. Create run directory:

```bash
RUN_DIR="$OUT/{example}/seed_{seed}"
mkdir -p "$RUN_DIR"
```

2. Execute command with timing:

```bash
/usr/bin/time -f "ELAPSED_SEC=%e" -o "$RUN_DIR/time.txt" \
  bash -lc "{command_template}" \
  >"$RUN_DIR/stdout.log" 2>"$RUN_DIR/stderr.log"
echo $? > "$RUN_DIR/exit_code.txt"
```

3. Parse `FINAL_METRIC` line and write `metrics.json`.

Expected metric line format in `stdout.log`:

```text
FINAL_METRIC: <metric_name>=<float>, threshold=<float|none>
```

If missing, set `status="missing_metric"`.

## Artifact Layout

```text
examples/neural/results/track_a/<RUN_ID>/
  run_manifest.json
  summary.csv
  summary.json
  comparisons/
    mnist_vs_deepproblog.json
    scallop_status.json
  01_minimal/
    seed_7/
      stdout.log
      stderr.log
      time.txt
      exit_code.txt
      metrics.json
    seed_42/...
    seed_123/...
  02_coins/...
  03_mnist_multidigit/...
  04_hwf/...
  05_poker/...
  06_clutrr/...
```

## JSON Schema (Per-Run `metrics.json`)

```json
{
  "track": "A",
  "run_id": "20260216T000000Z_track_a_dev",
  "example": "02_coins",
  "seed": 42,
  "status": "ok",
  "exit_code": 0,
  "duration_sec": 12.34,
  "command": "python examples/neural/02_coins/train.py ...",
  "metric": {
    "name": "test_acc",
    "value": 0.5000,
    "threshold": null
  },
  "dataset": {
    "manifest_path": "examples/neural/02_coins/dataset.json",
    "manifest_sha256": "<sha256>",
    "completeness": "provisional"
  },
  "environment": {
    "python": "3.10.x",
    "torch": "2.x+cu128",
    "pyxlog": "0.4.0",
    "gpu_name": "NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU",
    "driver": "573.71",
    "cuda_visible": true
  },
  "git": {
    "branch": "coins-alpha-gate",
    "commit": "<sha>"
  }
}
```

## Aggregate Schema (`summary.csv`)

Columns:

```text
run_id,example,seed,status,exit_code,duration_sec,metric_name,metric_value,metric_threshold,gate_pass
```

`gate_pass` rules:

- `true` if threshold is `null` or metric >= threshold.
- `false` if threshold present and metric < threshold.
- empty if metric missing.

## Aggregate Schema (`summary.json`)

```json
{
  "track": "A",
  "run_id": "20260216T000000Z_track_a_dev",
  "seed_list": [7, 42, 123],
  "examples": {
    "01_minimal": {
      "metric_name": "heldout_addition_acc",
      "n": 3,
      "mean": 0.0,
      "std": 0.0,
      "min": 0.0,
      "max": 0.0
    }
  },
  "notes": [
    "Track A hardware is not RTX 3090; timing is provisional.",
    "Scallop comparison deferred (not installed).",
    "Data completeness for 02/03/04 is provisional."
  ]
}
```

## Comparison Artifacts

### `comparisons/mnist_vs_deepproblog.json`

Fields:

- `xlog_track_a`: mean/std/time for `01_minimal`.
- `deepproblog_reference`: values sourced from `docs/reports/2026-02-10-deepproblog-baseline-gpu-sequential.md`.
- `protocol_match`: boolean.
- `comparison_scope`: `"provisional"` for Track A.

### `comparisons/scallop_status.json`

Fields:

- `available`: `false`
- `reason`: `"scallopy/scallop not installed in environment"`
- `deferred_to`: `"Track B"`

## Completion Criteria (Track A)

- All 16 runs complete (01_minimal: 1 seed; 02-06: 3 seeds each).
- Every run has `stdout.log`, `stderr.log`, `time.txt`, `exit_code.txt`, `metrics.json`.
- `summary.csv` and `summary.json` generated with no missing rows.
- Comparison artifacts generated (real data only, no placeholders):
  - `mnist_vs_deepproblog.json` — blocked if 01_minimal has no metric
  - `scallop_status.json` — blocked (not installed)
- No claims of charter completion attached to Track A outputs.

## Track A to Track B Handoff

Track A artifacts must explicitly mark:

- `hardware_reference_compliant=false` (not RTX 3090).
- `scallop_comparison_complete=false`.
- `dataset_finalized=false` for any provisional datasets.

Track B will re-run with final datasets + RTX 3090 + Scallop parity.

