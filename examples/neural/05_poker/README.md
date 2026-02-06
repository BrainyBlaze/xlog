# 05_poker

Real playing-card recognition example.

Expected layout:
- `data/cards/**/*.jpg` (or `.png`)
- Filenames should encode rank+suit, e.g. `AS.jpg`, `10H.png`, `QC.jpg`

Run:
- `python train.py --mode ci`
- `python train.py --mode dev`
- `python train.py --mode release`
