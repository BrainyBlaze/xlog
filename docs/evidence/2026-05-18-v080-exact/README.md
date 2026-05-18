# v0.8.0 Native Exact-Induction Evidence

**Date:** 2026-05-18
**Branch:** `feat/v080-dts-ml-python-productization`
**Scope:** G080_EXACT native exact-induction downstream consumer integration.

## Artifacts

| Artifact | Purpose |
|----------|---------|
| `runtime_probe.json` | Branch-local pyxlog probe for native-vs-strict-Python parity, D2H scaling, packaged `ilp_exact` kernels, and DTS liveness evidence anchors. |
| `python/tests/test_v080_exact_source.py` | Source/evidence guard for G080_EXACT contracts. |
| `python/tests/test_ilp_exact_induce.py` | Runtime parity and D2H-scaling contract for `induce_exact(..., backend="native")`. |
| `docs/architecture/bounded-exact-induction.md` | Type-dispatch and packaging policy. |
| `docs/architecture/python-bindings.md` | Public Python API notes for bounded exact induction. |
| `ROADMAP.md` | v0.8.0 native exact checklist status. |

## Validation Commands

| Command | Result |
|---------|--------|
| `/tmp/xlog-v080-cert-venv/bin/python -m pytest -q python/tests/test_v080_exact_source.py python/tests/test_ilp_exact_induce.py` | exit 0; 5 passed |
| `/tmp/xlog-v080-cert-venv/bin/python -m pytest -q python/tests/test_ilp_exact_induce.py` | exit 0; 2 passed |
| `cargo test -p xlog-induce --lib` | exit 0; 23 passed |
| `cargo test -p xlog-cuda --lib ilp_exact` | exit 0; 3 passed |
| `PYTHONPATH=/home/dev/projects/dts-dlm/src /tmp/xlog-v080-cert-venv/bin/python -m pytest -q /home/dev/projects/dts-dlm/src/tests/integrations/pyxlog/test_request_hash.py /home/dev/projects/dts-dlm/src/tests/integrations/test_tensorized_ilp.py -k 'exact_candidate_relations or request_hash'` | exit 0; 4 passed, 28 deselected |
| `/tmp/xlog-v080-cert-venv/bin/python` branch-local exact probe | exit 0; wrote `runtime_probe.json` values |

The runtime probe used branch-local `pyxlog` `0.7.0`, PyTorch
`2.10.0+cu128`, and CUDA device `NVIDIA RTX PRO 3000 Blackwell Generation
Laptop GPU`.

## Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M080_EXACT.1 consumer path | downstream tensorized ILP calls native backend without private hooks | PASS | DTS-DLM consumer `dts_dlm.integrations.pyxlog.tensorized_ilp` imports public `pyxlog.ilp.exact_induce.induce_exact` and dispatches `backend="native"`; xlog source guards pin the public pyxlog surface. |
| M080_EXACT.2 liveness | 449/449 benchmark reproduced | PASS | Accepted DTS-DLM evidence `/home/dev/projects/dts-dlm/docs/evidence/2026-04-17-m8-phase1-engine-integration.md` records native `both_heads_alive=449/449`, head 0 alive `449/449`, and head 1 alive `449/449`. This v080 branch re-ran the native exact parity and D2H gates, but did not relaunch the full DTS 449-doc job. |
| M080_EXACT.3 safety gates | rollback and quarantine rates unchanged from accepted baseline | PASS | Same DTS evidence records `rollback_rate=0.0` and `quarantine_rate=0.0`; `runtime_probe.json` records both anchors as present. |
| M080_EXACT.4 type dispatch | `U64` retained; `U32` and `Symbol` supported or explicitly deferred with evidence | PASS | `U64` runtime probe passed. `U32` and `Symbol` exact-induction dispatch are explicitly deferred in `docs/architecture/bounded-exact-induction.md`; `xlog-induce` currently validates exact buffers as `U64`. |
| M080_EXACT.5 packaging | committed PTX or documented no-PTX policy aligned with ILP-family convention | PASS | `ilp_exact` is in the CUDA kernel manifest; branch-local package contains `ilp_exact.portable.ptx` and `ilp_exact.sm_120.cubin`; docs record no checked-in generated PTX and generated portable PTX staging/embedding. |

## Runtime Probe Summary

Native exact induction matched the strict Python reference on the bounded U64
fixture:

- `native_total_scored_n3=36`
- `native_candidate_count_n3=3`
- `native_candidate_count_output_n3=6`
- first candidate: `chain p_A :- p_B, p_C`, `positives_covered=2`, `negatives_covered=0`

D2H scaling remained flat:

- small request: `candidate_count=2`, `total_scored=16`, `dtoh_calls=2`
- large request: `candidate_count=5`, `total_scored=100`, `dtoh_calls=2`
- gate: `large_dtoh_calls <= small_dtoh_calls + 2`

## Policy Notes

- Strict native exact induction scores each topology independently. The Python
  reference must be called with `strict_per_topology=True` for semantic parity.
- Legacy Python prototype behavior remains available with
  `strict_per_topology=False`; that mode preserves stale-mask contamination for
  historical compatibility and is not the native semantics.
- `U32` and `Symbol` exact dispatch are deferred until a named downstream
  consumer needs them. The eventual implementation should add explicit
  width/type dispatch rather than narrowing through the current `U64` kernel.
- Kernel source stays as `kernels/ilp_exact.cu`. Generated PTX/cubin artifacts
  are built, staged into `pyxlog/kernels/`, and checked by the install helper;
  generated PTX is not checked into git.

No push, tag, release-board update, merge, or final v0.8.0 closure claim is
authorized by this evidence.
