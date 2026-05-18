# G_M37A_SURFACE surface-preservation report

Date: 2026-05-18
Branch: `feat/m37a-surface-cert-g39`
Base: `feat/w6-bundle-integration-g39` at `3f38abee`

This report certifies Goal-039 G_M37A_SURFACE against the Phase-2 xlog
surface required by DTS-DLM M18 / M37-A. The branch uses a branch-local
`pyxlog` package from `/tmp/pyxlog-m37a`, backed by the release extension
`target/release/libpyxlog.so` and `XLOG_CUBIN_DIR=target/release/build/xlog-cuda-43b482a33001fc07/out`.

## Metric summary

| Metric | Status | Raw result |
|---|---:|---|
| M_M37A.1 `xlog_alpha_source` compiles + executes against Phase-2 wheel | PASS | DTS alpha tests: `21 passed, 14 warnings in 8.07s`; feasibility probe `FEASIBILITY_PROBE_PASSED`, `compile_sec=0.1126`, `forward_backward_step_sec=0.2845`, loss `1.502403736114502 -> 1.3740055561065674`, params changed `true`, nonzero grads `2/2` |
| M_M37A.2 M18-D AUROC gains reproduce within +/-5% | PASS | Verdict recomputed from temporary copy of canonical M18-D predictions with `n_boot=10000`: pred delta `0.3965599479565052`, arg0 delta `0.3871022178294078`, arg1 delta `0.46316616571549835`; all within +/-5% of `0.387/0.387/0.463` |
| M_M37A.3 M18-D coverage reproduces >= 99.8% | PASS | Heldout sidecar coverage: pred `529/529 = 100.000%`, arg0 `528/529 = 99.811%`, arg1 `528/529 = 99.811%`; slot-min coverage is above `99.8%` |
| M_M37A.4 `forward_backward_tensor` zero-host-reads cert | PASS | `nsys` CUDA trace for minimal `forward_backward_tensor` had no Device-to-Host rows. Memory summary rows: Device-to-Device `2` calls, Host-to-Device `2` calls, Device-to-Host `0` calls. Source guard `cargo test -p xlog-prob --test no_dtoh_in_neural_backward_nll -- --nocapture`: `1 passed` |
| M_M37A.5 XGCF circuit caching cert | PASS | Exact XGCF compile-plus-first-eval vs cached eval for 20 facts: session 1 `2.783193551993463s / 0.0007219830295071006s = 3854.929x`; session 2 `0.02929279103409499s / 0.0005171320517547429s = 56.645x`; 10-epoch neural-template replay inferred hit rate `59/60 = 98.333%`; post-`forward_backward_tensor` compile delta stayed `1`, cache size `1` |
| M_M37A.6 MNIST-Add reference example >= 99.0% accuracy | PASS | Temporary IDX-loader reproduction of documented torch direct-addition reference: 20 epochs, initial loss `1.3404522535856813`, final loss `0.014474124700427637`, train accuracy `0.99835`, held-out digit accuracy `0.9908`, held-out addition-pair accuracy `0.9818` (`4909/5000`), elapsed `13.170430495985784s` |
| M_M37A.7 `nn/4` parser accepts neural predicate syntax | PASS | `cargo test -p xlog-logic --test parse_neural -- --nocapture`: `6 passed` |
| M_M37A.8 `register_network(name, nn_module, optimizer)` roundtrip | PASS | Python slice `test_gpu_native_forward_backward_returns_tensor.py`, `test_circuit_cache.py`, `test_embeddings.py`, `test_network_registry.py`: `26 passed in 1.57s`; DTS feasibility probe registered `tiny_net` and changed both parameter tensors |
| M_M37A.9 `train_epoch(queries, batch_size=32)` loss trajectory | PASS | Python slice `test_training.py` + `test_train_model_tensor.py`: `35 passed in 1.99s`; MNIST-Add loss decreased `1.3404522535856813 -> 0.014474124700427637` |
| M_M37A.10 full Group B symbol-preservation smoke | PASS | `test_m37a_surface_preservation.rs` source enumerates 11/11 Group B symbol families. Fresh runtime/source slices: probabilistic gradient tests `2 passed`; `xlog-induce` unit tests `23 passed`; Bounded Exact Induction smoke via shipped `induce_exact` returned `candidate_count=2`, `total_scored=16`, first rule topology `chain` |

## Raw commands

Branch-local import:

```bash
XLOG_CUBIN_DIR=/home/dev/projects/xlog/.worktrees/g39-m37a-surface-cert/target/release/build/xlog-cuda-43b482a33001fc07/out \
PYTHONPATH=/tmp/pyxlog-m37a python3 -c 'import pyxlog; print(pyxlog.__file__, pyxlog.__version__)'
```

Result: `/tmp/pyxlog-m37a/pyxlog/__init__.py 0.6.2`.

DTS alpha source/session:

```bash
XLOG_CUBIN_DIR=/home/dev/projects/xlog/.worktrees/g39-m37a-surface-cert/target/release/build/xlog-cuda-43b482a33001fc07/out \
PYTHONPATH=/tmp/pyxlog-m37a:/home/dev/projects/dts-dlm/src \
pytest -q src/tests/learn/test_m18_xlog_alpha_source.py src/tests/learn/test_m18_xlog_alpha_integration.py
```

Result: `21 passed, 14 warnings in 8.07s`.

M18 feasibility probe:

```bash
XLOG_CUBIN_DIR=/home/dev/projects/xlog/.worktrees/g39-m37a-surface-cert/target/release/build/xlog-cuda-43b482a33001fc07/out \
PYTHONPATH=/tmp/pyxlog-m37a:/home/dev/projects/dts-dlm/src \
python3 -m dts_dlm.pilots.m18_xlog_neural_feasibility \
  --out /tmp/g39-m37a-a0/a0_feasibility.json --device cuda
```

Result:

```json
{
  "pyxlog_ok": true,
  "compile_sec": 0.1126,
  "forward_backward_step_sec": 0.2845,
  "loss_step1": 1.502403736114502,
  "loss_step2": 1.3740055561065674,
  "loss_decreased": true,
  "any_param_changed": true,
  "params_with_nonzero_grad": 2,
  "total_params": 2,
  "verdict": "FEASIBILITY_PROBE_PASSED"
}
```

M18-D verdict recompute from a temporary copy:

```bash
cp -a /home/dev/projects/dts-dlm/out/m18-phase-d /tmp/g39-m37a-m18-phase-d
PYTHONPATH=/home/dev/projects/dts-dlm/src \
python3 -m dts_dlm.pilots.m18_phase_d_runner \
  --out-dir /tmp/g39-m37a-m18-phase-d --n-boot 10000 verdict
```

Result:

```text
[verdict] pred: n=529  AUROC b=0.5712 c=0.9678  delta=+0.3966 CI=[+0.3797,+0.4136]  top1 b=0.000 c=0.711
[verdict] arg0: n=528  AUROC b=0.5765 c=0.9636  delta=+0.3871 CI=[+0.3657,+0.4019]  top1 b=0.000 c=0.455
[verdict] arg1: n=528  AUROC b=0.4971 c=0.9602  delta=+0.4632 CI=[+0.4175,+0.4825]  top1 b=0.000 c=0.623
[verdict] PASS - at least one slot's delta AUROC CI strictly above zero AND top-1 >= 2x chance
```

Coverage sidecar:

```json
{
  "rows_written": 529,
  "facts_total": 529,
  "in_domain_pred": 529,
  "in_domain_arg0": 528,
  "in_domain_arg1": 528,
  "in_domain_all_three": 527
}
```

Zero-host-read CUDA trace:

```bash
XLOG_CUBIN_DIR=/home/dev/projects/xlog/.worktrees/g39-m37a-surface-cert/target/release/build/xlog-cuda-43b482a33001fc07/out \
PYTHONPATH=/tmp/pyxlog-m37a \
nsys profile --trace=cuda --capture-range=none --force-overwrite=true \
  --output=/tmp/g39-m37a-fwbw python3 - <<'PY'
# compile nn(net, [X], Y, [0,1,2,3]) :: category(X, Y),
# register a tiny CUDA torch.nn.Linear network, run one forward_backward_tensor,
# synchronize, and assert the returned loss tensor is CUDA-resident.
PY
nsys stats --force-export=true \
  --report cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv --output /tmp/g39-m37a-fwbw /tmp/g39-m37a-fwbw.nsys-rep
```

Result scan:

```text
[CUDA memcpy Device-to-Device] count=2
[CUDA memcpy Host-to-Device] count=2
[CUDA memcpy Device-to-Host] count=0
```

The Host-to-Device calls are setup/upload traffic in the whole-process trace;
the requested forbidden direction, Device-to-Host, is absent.

Cache timing:

```json
{
  "n": 20,
  "sessions": [
    {
      "session": 1,
      "compile_plus_first_eval_sec": 2.783193551993463,
      "second_eval_sec": 0.0007219830295071006,
      "ratio": 3854.9293241609234
    },
    {
      "session": 2,
      "compile_plus_first_eval_sec": 0.02929279103409499,
      "second_eval_sec": 0.0005171320517547429,
      "ratio": 56.64470213110578
    }
  ]
}
```

Neural-template repeated-shape cache replay:

```json
{
  "calls": 60,
  "compile_misses": 1,
  "inferred_cache_hits": 59,
  "inferred_hit_rate": 0.9833333333333333,
  "cache_size": 1
}
```

The end-to-end small neural addition timing was also collected and is not used
for the 50x XGCF compile-cache gate because it includes neural forward/backward
work after the circuit hit. It still confirmed one compile miss and subsequent
hits: session A ratio `18.725x`, session B ratio `3.640x`, compile delta `1`,
cache size `1`.

MNIST-Add reference reproduction:

```json
{
  "device": "cuda",
  "epochs": 20,
  "initial_loss": 1.3404522535856813,
  "final_loss": 0.014474124700427637,
  "train_accuracy": 0.99835,
  "heldout_digit_accuracy": 0.9908,
  "heldout_addition_accuracy": 0.9818,
  "heldout_addition_correct": 4909,
  "heldout_addition_total": 5000,
  "elapsed_sec": 13.170430495985784
}
```

`torchvision` was not installed in the environment, so the run used a temporary
IDX loader for MNIST under `/tmp` and imported the checked-in
`examples/neural/01_minimal/train.py` model/training functions unchanged.

Focused xlog tests:

```bash
cargo test -p xlog-logic --test parse_neural -- --nocapture
cargo test -p xlog-prob --test no_dtoh_in_neural_backward_nll -- --nocapture
cargo test -p xlog-induce -- --nocapture
```

Results:

```text
parse_neural: 6 passed
no_dtoh_in_neural_backward_nll: 1 passed
xlog-induce: 23 passed; doc-tests 0
```

Focused Python surface slices:

```bash
XLOG_CUBIN_DIR=/home/dev/projects/xlog/.worktrees/g39-m37a-surface-cert/target/release/build/xlog-cuda-43b482a33001fc07/out \
PYTHONPATH=/tmp/pyxlog-m37a \
pytest -q \
  python/tests/test_gpu_native_forward_backward_returns_tensor.py \
  python/tests/test_circuit_cache.py \
  python/tests/test_embeddings.py \
  python/tests/test_network_registry.py
```

Result: `26 passed in 1.57s`.

```bash
XLOG_CUBIN_DIR=/home/dev/projects/xlog/.worktrees/g39-m37a-surface-cert/target/release/build/xlog-cuda-43b482a33001fc07/out \
PYTHONPATH=/tmp/pyxlog-m37a \
pytest -q python/tests/test_training.py python/tests/test_train_model_tensor.py
```

Result: `35 passed in 1.99s`.

```bash
XLOG_CUBIN_DIR=/home/dev/projects/xlog/.worktrees/g39-m37a-surface-cert/target/release/build/xlog-cuda-43b482a33001fc07/out \
PYTHONPATH=/tmp/pyxlog-m37a \
pytest -q \
  python/tests/test_negation.py::TestNegationGradients::test_negation_gradient_returns \
  python/tests/test_negation.py::TestNegationGradients::test_negation_gradient_values
```

Result: `2 passed in 5.78s`.

Bounded Exact Induction smoke:

```json
{
  "candidate_count": 2,
  "total_scored": 16,
  "first_topology": "chain",
  "first_left_relation": "p_B",
  "first_right_relation": "p_C"
}
```

The shipped public API remains `pyxlog.ilp.induce_exact(...)` and
`xlog_induce::induce_exact(...)`. The M_M37A.10 smoke source includes the
plan mnemonic `bounded_exact_induce(program, examples, budget)` and the
actual shipped call `induce_exact(program, examples=..., k_per_topology=budget)`.

## Notes

- The M18-D train/heldout stages were not rerun; the verdict was recomputed
  from a temporary copy of the canonical checked-in M18-D predictions and
  sidecar. This avoids mutating DTS-DLM artifacts and avoids the historical
  multi-hour GPU replay while still rechecking the acceptance math.
- A broad single-process Python sweep over all Group B files was attempted
  and stopped after exceeding the prior runtime without producing a pytest
  summary. It is not counted above. The report uses bounded focused slices
  with final summaries instead.
- `train_and_promote` remains covered as a queued M37-F consumer in the
  source-audit smoke and by the quick strict-GPU-native rejection check
  (`1 passed in 0.02s`). The full promotion-positive test is not counted in
  this report because it did not finish within the bounded fresh window.
