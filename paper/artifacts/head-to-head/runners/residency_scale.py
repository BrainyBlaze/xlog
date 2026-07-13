"""CRIT-2 at-scale residency ablation (batched path).

The single-query residency sweep measured only 2 neural->symbolic handoffs per
iteration. Real training batches many queries per step, so the host round-trip a
CPU-reasoning hybrid pays scales with the batch. Here we drive the *batched*
path (forward_backward_grouped over N addition queries -> N*2 handoffs) and sweep
N, with vs without XLOG_FORCE_HOST_ROUNDTRIP, to measure how the transfer tax
grows with scale. Compile once, reuse (toggle the env per measurement) so we pay
the ~40 s CDCL verify only once. lr=0 keeps weights fixed (timing-only).
The per-buffer round-trip is an upper bound on a hybrid's coalesced transfer.
"""
import os, time, json, statistics as st, torch, pyxlog

print("BOOT cuda=%s" % torch.cuda.is_available(), flush=True)

SRC = ("nn(net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).\n"
       "addition(I, J, S) :- digit(I, A), digit(J, B), S is A + B.\n")
NQ = (1, 4, 16, 64, 256)
MAXQ = max(NQ)
ITERS = 50


class Net(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = torch.nn.Linear(1, d)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


def queries(nq):
    qs, exp = [], []
    for k in range(nq):
        qs.append("addition(%d, %d, 4)" % (2 * k, 2 * k + 1))
        exp.append(True)
    return qs, exp


assert torch.cuda.is_available(), "CUDA required"
dev = torch.device("cuda")
torch.manual_seed(1)
prog = pyxlog.Program.compile(SRC)
print("compiled", flush=True)
net = Net(10).to(dev)
opt = torch.optim.SGD(net.parameters(), lr=0.0)
prog.register_network("net", net, opt, batching=True)
prog.add_tensor_source("data", torch.randn(2 * MAXQ, 1, device=dev))
print("registered", flush=True)


def measure(nq, force):
    if force:
        os.environ["XLOG_FORCE_HOST_ROUNDTRIP"] = "1"
    else:
        os.environ.pop("XLOG_FORCE_HOST_ROUNDTRIP", None)
    qs, exp = queries(nq)
    for _ in range(3):
        prog.forward_backward_grouped(qs, exp)
    torch.cuda.synchronize()
    t0 = time.monotonic()
    for _ in range(ITERS):
        prog.forward_backward_grouped(qs, exp)
    torch.cuda.synchronize()
    return (time.monotonic() - t0) / ITERS * 1000.0


rows = []
for nq in NQ:
    offs = [measure(nq, False) for _ in range(3)]
    ons = [measure(nq, True) for _ in range(3)]
    off_m, on_m = st.mean(offs), st.mean(ons)
    row = {
        "queries": nq, "handoffs": 2 * nq,
        "off_ms": round(off_m, 4), "on_ms": round(on_m, 4),
        "roundtrip_ms": round(on_m - off_m, 4),
        "overhead_pct": round(100 * (on_m - off_m) / off_m, 1) if off_m > 0 else None,
        "roundtrip_per_handoff_us": round((on_m - off_m) / (2 * nq) * 1000, 2) if nq else None,
    }
    rows.append(row)
    print("ROW " + json.dumps(row), flush=True)

json.dump(rows, open("/workspace/residency_scale.json", "w"), indent=2)
print("RESIDENCY_SCALE_DONE rows=%d" % len(rows), flush=True)
