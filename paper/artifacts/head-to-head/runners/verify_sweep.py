"""EIC W5 — isolate CDCL-verify vs D4-compile across circuit sizes.

Linear probabilistic reachability chains (no branching -> stays compilable);
fresh process so every compile is a real cache miss. Reads the per-stage cold
profile via warmup_breakdown().
"""
import os
os.environ["XLOG_WARMUP_PROFILE"] = "1"
import json, pyxlog


def prog(n):
    lines = [f"0.5::edge({i},{i+1})." for i in range(n)]
    lines.append("path(X,Y) :- edge(X,Y).")
    lines.append("path(X,Y) :- edge(X,Z), path(Z,Y).")
    lines.append(f"query(path(0,{n})).")
    return "\n".join(lines)


rows = []
for n in (5, 10, 15, 20, 30, 40, 50, 60):
    p = pyxlog.Program.compile(prog(n))
    c = p.warmup_breakdown()["circuit"]
    cold = (c["d4_compile_sec"] + c["verify_sec"] + c["smooth_sec"]
            + c["cache_store_sec"] + c["cnf_hash_sec"] + c["free_var_mask_sec"])
    row = {
        "n": n,
        "d4_compile_ms": round(c["d4_compile_sec"] * 1000, 3),
        "verify_ms": round(c["verify_sec"] * 1000, 3),
        "cold_ms": round(cold * 1000, 3),
        "verify_pct_of_cold": round(100 * c["verify_sec"] / cold, 1) if cold > 0 else None,
        "gpu_cache_hit": c["gpu_cache_hit"],
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

json.dump(rows, open("/workspace/verify_sweep.json", "w"), indent=2)
print("VERIFY_SWEEP_DONE rows=%d" % len(rows), flush=True)
