# 03_mnist_multidigit

Real SVHN two-digit recognition example.

Expected layout:
- `data/svhn/train/digitStruct.mat`
- `data/svhn/train/*.png`

The loader filters SVHN to 2-digit samples and trains `digit_left/2` and `digit_right/2`.

Run:
- `python train.py --mode ci`
- `python train.py --mode dev`
- `python train.py --mode release`
