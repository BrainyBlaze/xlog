# Python Examples (`xlog-gpu`)

These scripts demonstrate the `xlog_gpu` Python module (built from `crates/xlog-gpu-py`) using **DLPack** for GPU table interchange.

## Build (wheel) locally

```bash
cd crates/xlog-gpu-py
python -m pip install --upgrade pip maturin
maturin develop --release
```

## Run

```bash
python examples/python/01_dlpack_reachability_torch.py
python examples/python/02_prob_wet_conditioning_torch.py
```

Torch is optional; the module accepts any DLPack producer (e.g., cuDF, CuPy, JAX).

