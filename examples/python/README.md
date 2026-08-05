# Python Examples (`pyxlog`)

These scripts demonstrate the `pyxlog` Python module (built from
`crates/pyxlog`) using **DLPack** for GPU table interchange. One-shot inputs and
the consumer handoff of exported tensor views are zero-copy. Persistent relation
replacements take an owned device-to-device snapshot, and exporting a stored
relation makes a device-to-device retention copy before handing off its buffers.

## Build (wheel) locally

```bash
cd crates/pyxlog
python -m pip install --upgrade pip maturin
maturin develop --release
```

## Run

```bash
python examples/python/01_dlpack_reachability_torch.py
python examples/python/02_prob_wet_conditioning_torch.py
python examples/python/03_prob_mc_nonmonotone_torch.py
```

Torch is optional; the module accepts CUDA-backed DLPack producers such as cuDF,
CuPy, and GPU-backed JAX arrays. Supply one 1-D producer object per relation
column rather than manufacturing raw capsules.
