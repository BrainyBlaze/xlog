#!/usr/bin/env python3
"""External Consumer Fit Benchmark: xlog induction on external consumer-shaped request surfaces.

Reproduces the exact workload an external consumer sends to xlog's train_on_compiled_relations():
- 1-3 predicates, 20-30 positives, 4 bootstrap negatives
- Four topology masks (chain, star, fanout, fanin)
- Repeated reruns on identical input
- Deterministic vs non-deterministic comparison

Usage:
    .venv/bin/python python/examples/dts_fit_benchmark.py

Outputs structured JSON to stdout for CI gating.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import torch


@dataclass
class BenchmarkResult:
    regime: str
    n_repeats: int
    n_preds: int
    n_pos: int
    n_neg: int
    deterministic: bool
    per_mask: dict = field(default_factory=dict)
    total_steps: list[int] = field(default_factory=list)
    total_rules: list[int] = field(default_factory=list)
    total_times: list[float] = field(default_factory=list)
    rule_fingerprints: list[str] = field(default_factory=list)

    @property
    def unique_fingerprints(self) -> int:
        return len(set(self.rule_fingerprints))

    @property
    def is_deterministic(self) -> bool:
        return self.unique_fingerprints <= 1

    def summary(self) -> dict:
        n = self.n_repeats
        return {
            "regime": self.regime,
            "deterministic_mode": self.deterministic,
            "n_preds": self.n_preds,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "mean_steps": sum(self.total_steps) / n if n else 0,
            "mean_rules": sum(self.total_rules) / n if n else 0,
            "rule_variance": max(self.total_rules) - min(self.total_rules) if n else 0,
            "mean_time": sum(self.total_times) / n if n else 0,
            "unique_fingerprints": self.unique_fingerprints,
            "is_deterministic": self.is_deterministic,
        }


def build_dts_program(pred_ids: list[int], target_name: str) -> str:
    """Build an external consumer-shaped Datalog program with four learnable topology masks."""
    lines = []
    for pid in pred_ids:
        lines.append(f"pred p_{pid}(u64, u64).")
    # Four canonical topologies — same as the external consumer _run_xlog_training
    lines.append(f"learnable(W_chain_{target_name}) :: {target_name}(X, Y) :- bL(X, Z), bR(Z, Y).")
    lines.append(f"learnable(W_star_{target_name}) :: {target_name}(X, Y) :- bL(X, Y), bR(X, Y).")
    lines.append(f"learnable(W_fanout_{target_name}) :: {target_name}(X, Y) :- bL(X, Z), bR(X, Y).")
    lines.append(f"learnable(W_fanin_{target_name}) :: {target_name}(X, Y) :- bL(X, Y), bR(Z, Y).")
    return "\n".join(lines) + "\n"


def generate_dts_facts(
    n_preds: int,
    n_pos: int,
    n_neg: int,
    n_entities: int = 30,
    seed: int = 42,
) -> tuple[list[int], int, dict, dict]:
    """Generate external consumer-shaped fact tensors with a planted discoverable rule.

    Plants a chain rule: target(X,Y) :- bL(X,Z), bR(Z,Y)
    Body predicates get random facts. Target positives are derived from the
    chain join on body facts. Negatives are entity pairs NOT derivable.

    Returns (pred_ids, target_pid, positive_facts, negative_facts).
    """
    import random
    rng = random.Random(seed)

    pred_ids = list(range(10000, 10000 + n_preds))
    target_pid = pred_ids[0]
    device = torch.device("cuda")
    entities = list(range(1, n_entities + 1))

    # Generate body predicate facts (random)
    positive_facts = {}
    for pid in pred_ids[1:]:  # skip target, generate body preds
        n_body = rng.randint(15, 25)
        pairs = set()
        while len(pairs) < n_body:
            pairs.add((rng.choice(entities), rng.choice(entities)))
        a0 = torch.tensor([p[0] for p in pairs], dtype=torch.int64, device=device)
        a1 = torch.tensor([p[1] for p in pairs], dtype=torch.int64, device=device)
        positive_facts[pid] = (a0, a1)

    # Derive target positives from chain join on first two body preds
    # target(X,Y) :- body1(X,Z), body2(Z,Y)
    if len(pred_ids) >= 3:
        bL_pid, bR_pid = pred_ids[1], pred_ids[2]
    elif len(pred_ids) == 2:
        bL_pid = bR_pid = pred_ids[1]
    else:
        # 1-predicate: target only, no body — degenerate case
        # Generate random positives (no rule to discover)
        pairs = set()
        while len(pairs) < n_pos:
            pairs.add((rng.choice(entities), rng.choice(entities)))
        a0 = torch.tensor([p[0] for p in pairs], dtype=torch.int64, device=device)
        a1 = torch.tensor([p[1] for p in pairs], dtype=torch.int64, device=device)
        positive_facts[target_pid] = (a0, a1)
        negative_facts = {target_pid: (
            torch.zeros(0, dtype=torch.int64, device=device),
            torch.zeros(0, dtype=torch.int64, device=device),
        )}
        return pred_ids, target_pid, positive_facts, negative_facts

    # Chain join: target(X,Y) = {(X,Y) : exists Z s.t. bL(X,Z) AND bR(Z,Y)}
    bL_pairs = set(zip(positive_facts[bL_pid][0].tolist(), positive_facts[bL_pid][1].tolist()))
    bR_pairs = set(zip(positive_facts[bR_pid][0].tolist(), positive_facts[bR_pid][1].tolist()))

    # Build Z-index for bL
    bL_by_z = {}
    for x, z in bL_pairs:
        bL_by_z.setdefault(z, []).append(x)

    derived = set()
    for z, y in bR_pairs:
        if z in bL_by_z:
            for x in bL_by_z[z]:
                derived.add((x, y))

    # Sample n_pos from derived (or all if fewer)
    derived_list = list(derived)
    rng.shuffle(derived_list)
    target_pairs = derived_list[:n_pos]
    if len(target_pairs) < n_pos:
        # Pad with more random derived pairs if too few
        while len(target_pairs) < n_pos and len(derived_list) > len(target_pairs):
            target_pairs = derived_list[:n_pos]
            break

    a0 = torch.tensor([p[0] for p in target_pairs], dtype=torch.int64, device=device)
    a1 = torch.tensor([p[1] for p in target_pairs], dtype=torch.int64, device=device)
    positive_facts[target_pid] = (a0, a1)

    # Generate negative facts: entity pairs NOT in derived set
    pos_set = set(target_pairs)
    neg_pairs = set()
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < n_neg * 100:
        p = (rng.choice(entities), rng.choice(entities))
        if p not in pos_set and p not in derived:
            neg_pairs.add(p)
        attempts += 1
    neg_a0 = torch.tensor([p[0] for p in neg_pairs], dtype=torch.int64, device=device)
    neg_a1 = torch.tensor([p[1] for p in neg_pairs], dtype=torch.int64, device=device)
    negative_facts = {target_pid: (neg_a0, neg_a1)}

    return pred_ids, target_pid, positive_facts, negative_facts


def run_benchmark(
    n_preds: int,
    n_pos: int,
    n_neg: int,
    n_repeats: int = 10,
    deterministic: bool = False,
    step_budget: int = 25,
    max_attempts: int = 1,
    global_step_limit: int = 50,
    seed: int = 42,
) -> BenchmarkResult:
    """Run the external consumer-fit benchmark: compile, upload, train four masks, repeat."""
    from pyxlog import IlpProgramFactory
    from pyxlog.ilp.trainer import train_on_compiled_relations
    from pyxlog.ilp.types import TrainConfig

    pred_ids, target_pid, positive_facts, negative_facts = generate_dts_facts(
        n_preds, n_pos, n_neg, seed=seed,
    )
    target_name = f"p_{target_pid}"
    source = build_dts_program(pred_ids, target_name)

    mask_names = [
        f"W_chain_{target_name}",
        f"W_star_{target_name}",
        f"W_fanout_{target_name}",
        f"W_fanin_{target_name}",
    ]

    config = TrainConfig(
        strict_gpu_native=True,
        max_mined_negatives=0,
        device=0,
        global_step_limit=global_step_limit,
        step_budget_per_attempt=step_budget,
        max_attempts=max_attempts,
        deterministic=deterministic,
        seed=seed if deterministic else None,
    )

    regime = f"{n_preds}p_{n_pos}pos_{n_neg}neg"
    if deterministic:
        regime += "_det"

    result = BenchmarkResult(
        regime=regime,
        n_repeats=n_repeats,
        n_preds=n_preds,
        n_pos=n_pos,
        n_neg=n_neg,
        deterministic=deterministic,
    )

    positives = {target_name: [positive_facts[target_pid][0], positive_facts[target_pid][1]]}
    negatives = {target_name: [negative_facts[target_pid][0], negative_facts[target_pid][1]]}

    for rep in range(n_repeats):
        # Compile fresh each repeat (matches external consumer: one compile per _run_xlog_training)
        prog = IlpProgramFactory.compile(source, device=0, memory_mb=128)

        # Upload all relation data
        for pid in pred_ids:
            if pid in positive_facts:
                a0, a1 = positive_facts[pid]
                if a0.shape[0] > 0:
                    prog.put_relation(f"p_{pid}", [a0, a1])

        # Train four masks separately (matches external consumer)
        all_rules = []
        total_steps = 0
        t0 = time.time()

        for mask_name in mask_names:
            mask_result = train_on_compiled_relations(
                prog, mask_name, positives, negatives, config,
            )
            steps = getattr(mask_result, "total_steps", 0)
            total_steps += steps
            # StrictTrainResult has singular discovered_rule; TrainResult may have plural
            rules = getattr(mask_result, "discovered_rules", None) or []
            if not rules:
                single = getattr(mask_result, "discovered_rule", None)
                if single:
                    rules = [single]
            all_rules.extend(str(r) for r in rules)

        dt = time.time() - t0
        fp = hashlib.md5("|".join(sorted(all_rules)).encode()).hexdigest()[:12]

        result.total_steps.append(total_steps)
        result.total_rules.append(len(all_rules))
        result.total_times.append(dt)
        result.rule_fingerprints.append(fp)

    return result


def main():
    print("=== EXTERNAL CONSUMER FIT XLOG BENCHMARK ===\n")

    configs = [
        # external consumer current regime
        {"n_preds": 3, "n_pos": 27, "n_neg": 4, "deterministic": False, "label": "external consumer current (3p, nondeterministic)"},
        {"n_preds": 3, "n_pos": 27, "n_neg": 4, "deterministic": True, "label": "external consumer + deterministic mode"},
        # Degenerate case
        {"n_preds": 1, "n_pos": 27, "n_neg": 4, "deterministic": False, "label": "Degenerate (1p)"},
        # Showcase-like
        {"n_preds": 6, "n_pos": 20, "n_neg": 20, "deterministic": False, "label": "Showcase-like (6p, balanced neg)"},
        {"n_preds": 6, "n_pos": 20, "n_neg": 20, "deterministic": True, "label": "Showcase-like + deterministic"},
    ]

    results = []
    for cfg in configs:
        label = cfg.pop("label")
        print(f"--- {label} ---")
        r = run_benchmark(n_repeats=10, **cfg)
        s = r.summary()
        results.append(s)
        print(f"  steps={s['mean_steps']:.0f}  rules={s['mean_rules']:.1f}  "
              f"var={s['rule_variance']}  time={s['mean_time']:.1f}s  "
              f"unique_fps={s['unique_fingerprints']}/10  "
              f"deterministic={s['is_deterministic']}")
        print()

    # Summary table
    print("=== SUMMARY ===\n")
    print(f"{'regime':<35} {'steps':>6} {'rules':>6} {'var':>4} {'time':>6} {'unique':>7} {'det?':>5}")
    print("-" * 75)
    for s in results:
        print(f"{s['regime']:<35} {s['mean_steps']:>6.0f} {s['mean_rules']:>6.1f} "
              f"{s['rule_variance']:>4} {s['mean_time']:>5.1f}s "
              f"{s['unique_fingerprints']:>5}/10 {'YES' if s['is_deterministic'] else 'NO':>5}")

    # Write JSON
    with open("/tmp/dts_fit_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /tmp/dts_fit_benchmark.json")


if __name__ == "__main__":
    main()
