# Python Examples (`pyxlog`)

These scripts demonstrate the `pyxlog` Python module (built from `crates/pyxlog`) using **DLPack** for GPU table interchange.

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

Torch is optional; the module accepts any DLPack producer (e.g., cuDF, CuPy, JAX).
