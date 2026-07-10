"""CRIT-2 residency ablation — instrumented, hang-visible version.

Prints a BEGIN line before every potentially-blocking run so a stall is
localized. Output goes straight to stdout (run with `python -u` -> file, no
grep pipe). Per-run errors are caught so one bad config can't wedge the sweep.
"""
import os, sys, time, json, statistics as st, torch, pyxlog

print("BOOT cuda=%s" % torch.cuda.is_available(), flush=True)


class Net(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = torch.nn.Linear(1, d)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


def make_src(L):
    labels = "[" + ",".join(str(i) for i in range(L)) + "]"
    return (f"nn(net, [X], Y, {labels}) :: digit(X, Y).\n"
            "addition(I, J, S) :- digit(I, A), digit(J, B), S is A + B.\n")


def run(L, force, seed, iters=40):
    tag = f"L={L} force={force} seed={seed}"
    print("BEGIN " + tag, flush=True)
    if force:
        os.environ["XLOG_FORCE_HOST_ROUNDTRIP"] = "1"
    else:
        os.environ.pop("XLOG_FORCE_HOST_ROUNDTRIP", None)
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    prog = pyxlog.Program.compile(make_src(L))
    print("  compiled " + tag, flush=True)
    net = Net(L).to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    prog.register_network("net", net, opt, batching=True)
    torch.manual_seed(seed + 1)
    prog.add_tensor_source("data", torch.randn(10, 1, device=dev))
    for _ in range(3):
        prog.forward_backward("addition(0, 1, 2)")
    torch.cuda.synchronize()
    print("  warm done " + tag, flush=True)
    t0 = time.monotonic()
    for _ in range(iters):
        prog.forward_backward("addition(0, 1, 2)")
    torch.cuda.synchronize()
    ms = (time.monotonic() - t0) / iters * 1000.0
    print("  END %s -> %.4f ms/iter" % (tag, ms), flush=True)
    return ms


assert torch.cuda.is_available(), "CUDA required"
rows = []
for L in (4, 10):
    offs = [run(L, False, s) for s in (1, 2, 3)]
    ons = [run(L, True, s) for s in (1, 2, 3)]
    off_m, on_m = st.mean(offs), st.mean(ons)
    row = {
        "labels": L,
        "off_ms_mean": round(off_m, 4), "off_ms_std": round(st.pstdev(offs), 4),
        "on_ms_mean": round(on_m, 4), "on_ms_std": round(st.pstdev(ons), 4),
        "roundtrip_ms": round(on_m - off_m, 4),
        "residency_overhead_pct": round(100 * (on_m - off_m) / off_m, 1) if off_m > 0 else None,
    }
    rows.append(row)
    print("ROW " + json.dumps(row), flush=True)

json.dump(rows, open("/workspace/residency_sweep.json", "w"), indent=2)
print("RESIDENCY_SWEEP_DONE rows=%d" % len(rows), flush=True)
