# 04_hwf

Real handwritten-formula recognition example using CROHME.

Expected layout:
- `data/crohme/**/*.inkml`
- `data/crohme/**/*.txt` (same stem as InkML, contains LaTeX)

Run:
- `python train.py --mode ci`
- `python train.py --mode dev`
- `python train.py --mode release`
