# 06_clutrr

Real relational reasoning example using CLUTRR.

Expected layout:
- `data/clutrr/train.jsonl`

Each JSONL row must include:
- `story` (or `text`)
- `target_relation` (or `label`) in `{parent, child, spouse, sibling}`

Run:
- `python train.py --mode ci`
- `python train.py --mode dev`
- `python train.py --mode release`
