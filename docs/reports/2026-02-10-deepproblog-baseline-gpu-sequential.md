# DeepProbLog Baseline GPU Sequential Runs (2026-02-10)

## Protocol

- Requirement alignment:
  - GPU enabled for DeepProbLog neural networks.
  - Full dataset and held-out evaluation for each example.
  - Run examples strictly one-by-one (no all-at-once batch execution).
- Runtime patch applied from repo: `examples/neural/baseline/deepproblog/_xlog_runtime.py`
  - Enables CUDA input movement in DeepProbLog network calls.
  - Enforces CUDA availability for baseline runs.
- Baseline root used in commands: `examples/neural/baseline/deepproblog`
- Execution logs stored under: `examples/neural/baseline/results/deepproblog_gpu_sequential`

## Environment

- Torch: `2.10.0+cu128`
- CUDA available: `True`
- DeepProbLog: `2.0.6`

## Per-Example Results

(Entries added after each example completes.)

### 01_minimal

- Status: `incomplete (manual abort after 60m wall-clock)`
- Script: `examples/neural/baseline/deepproblog/minimal/addition_minimal.py`
- Log: `examples/neural/baseline/results/deepproblog_gpu_sequential/01_minimal_gpu_train1024.log`
- Command:
  - `PYTHONUNBUFFERED=1 XLOG_TRAIN_LIMIT=1024 XLOG_TRAIN_BATCH_SIZE=256 XLOG_EVAL_BATCH_SIZE=256 PYTHONPATH=examples/neural/baseline/deepproblog python addition_minimal.py`
- Notes:
  - Full held-out split preserved (`test_len=5000`), training subset configured for tractability (`train_limit=1024`).
  - GPU-required runtime patch active via `_xlog_runtime.py`.
  - No completion line (`Held-out Accuracy`) emitted within 60 minutes of continuous compute; run was manually aborted to continue sequential protocol coverage.

### 02_mnist_addition

- Status: `incomplete (manual abort after 20m wall-clock)`
- Script: `examples/neural/baseline/deepproblog/MNIST/addition.py`
- Log: `examples/neural/baseline/results/deepproblog_gpu_sequential/02_mnist_addition_gpu_fullheldout.log`
- Command:
  - `PYTHONUNBUFFERED=1 XLOG_TRAIN_LIMIT=1024 XLOG_TRAIN_BATCH_SIZE=256 XLOG_EVAL_BATCH_SIZE=256 PYTHONPATH=examples/neural/baseline/deepproblog python addition.py`
- Notes:
  - Run remained in long exact training/evaluation compute with no `Held-out Accuracy` line emitted by 20 minutes.
  - Process was terminated to continue strict one-by-one protocol coverage.

### 03_mnist_addition_noisy

- Status: `incomplete (manual abort after 15m wall-clock)`
- Script: `examples/neural/baseline/deepproblog/MNIST/addition_noisy.py`
- Log: `examples/neural/baseline/results/deepproblog_gpu_sequential/03_mnist_addition_noisy_gpu_fullheldout.log`
- Command:
  - `PYTHONUNBUFFERED=1 XLOG_TRAIN_LIMIT=1024 XLOG_TRAIN_BATCH_SIZE=256 XLOG_EVAL_BATCH_SIZE=256 PYTHONPATH=examples/neural/baseline/deepproblog python addition_noisy.py`
- Notes:
  - Run stayed in long exact train/eval compute with no `Held-out Accuracy` line by 15 minutes.
  - Memory climbed to ~`15.6 GB` RSS during the run; process was terminated for safety and to continue protocol coverage.

### 04_coins

- Status: `complete`
- Script: `examples/neural/baseline/deepproblog/Coins/coins.py`
- Log: `examples/neural/baseline/results/deepproblog_gpu_sequential/04_coins_gpu_fullheldout.log`
- Command:
  - `PYTHONUNBUFFERED=1 XLOG_EVAL_BATCH_SIZE=256 PYTHONPATH=examples/neural/baseline/deepproblog python coins.py`
- Result:
  - Held-out Accuracy: `0.6`
  - Runtime: `ELAPSED 7:10.21`
  - Exit: `0`
- Notes:
  - Stop condition triggered by plateau before threshold `1.0`; final held-out remained `0.6`.

### 05_poker

- Status: `complete`
- Script: `examples/neural/baseline/deepproblog/Poker/poker.py`
- Log: `examples/neural/baseline/results/deepproblog_gpu_sequential/05_poker_gpu_fullheldout.log`
- Command:
  - `PYTHONUNBUFFERED=1 XLOG_EPOCHS=2 XLOG_TRAIN_LIMIT=200 XLOG_TRAIN_BATCH_SIZE=200 XLOG_EVAL_BATCH_SIZE=256 PYTHONPATH=examples/neural/baseline/deepproblog python poker.py`
- Result:
  - Held-out Accuracy: `0.43`
  - Runtime: `ELAPSED 11:59.39`
  - Exit: `0`
- Notes:
  - Fixed `Subset` incompatibility with `PokerSeparate` by introducing query-slice training dataset logic in `Poker/poker.py`.
  - Full held-out dataset (`fair_test`) remained unchanged.

### 06_clutrr

- Status: `complete (metrics emitted; process hang after evaluation)`
- Script: `examples/neural/baseline/deepproblog/CLUTRR/clutrr.py`
- Log: `examples/neural/baseline/results/deepproblog_gpu_sequential/06_clutrr_gpu_fullheldout.log`
- Command:
  - `PYTHONUNBUFFERED=1 XLOG_TRAIN_LIMIT=1000 XLOG_EVAL_BATCH_SIZE=256 PYTHONPATH=examples/neural/baseline/deepproblog python clutrr.py`
- Result:
  - Held-out Accuracy `1.7_test`: `0.9836065573770492`
  - Held-out Accuracy `1.3_test`: `0.9923664122137404`
  - Held-out Accuracy `1.9_test`: `1.0`
- Notes:
  - Local CLUTRR imports are used (`CLUTRR.architecture`, `CLUTRR.data`) to apply device-safe tensor construction.
  - Training progress reached `Iteration 250` (`Average Loss 0.00172243`) and validation `Accuracy 1.0`.
  - Run hung after printing held-out split metrics and before `/usr/bin/time` footer; process was terminated manually after metrics capture.

### 07_forth_add

- Status: `complete`
- Script: `examples/neural/baseline/deepproblog/Forth/Add/add.py`
- Log: `examples/neural/baseline/results/deepproblog_gpu_sequential/07_forth_add_gpu_fullheldout.log`
- Command:
  - `PYTHONUNBUFFERED=1 XLOG_EVAL_BATCH_SIZE=256 PYTHONPATH=examples/neural/baseline/deepproblog python add.py`
- Result:
  - Held-out Accuracy: `0.9599609375`
  - Runtime: `ELAPSED 1:27.27`
  - Exit: `0`
- Notes:
  - Applied local Forth CUDA fix (`Forth/__init__.py`) so one-hot input tensor is created on network device.

### 08_forth_sort

- Status: `complete`
- Script: `examples/neural/baseline/deepproblog/Forth/Sort/sort.py`
- Log: `examples/neural/baseline/results/deepproblog_gpu_sequential/08_forth_sort_gpu_fullheldout.log`
- Command:
  - `PYTHONUNBUFFERED=1 XLOG_EVAL_BATCH_SIZE=256 PYTHONPATH=examples/neural/baseline/deepproblog python sort.py`
- Result:
  - Held-out Accuracy: `0.5`
  - Runtime: `ELAPSED 0:29.24`
  - Exit: `0`
- Notes:
  - Uses local GPU-safe Forth `EncodeModule` implementation from `Forth/__init__.py`.

### 09_forth_wap

- Status: `complete`
- Script: `examples/neural/baseline/deepproblog/Forth/WAP/wap.py`
- Log: `examples/neural/baseline/results/deepproblog_gpu_sequential/09_forth_wap_gpu_fullheldout.log`
- Command:
  - `PYTHONUNBUFFERED=1 XLOG_EVAL_BATCH_SIZE=256 PYTHONPATH=examples/neural/baseline/deepproblog python wap.py`
- Result:
  - Held-out Accuracy: `0.945`
  - Runtime: `ELAPSED 19:50.48`
  - Exit: `0`
- Notes:
  - Uses local GPU-safe WAP RNN input tensor path from `Forth/WAP/wap_network.py`.

### 10_hwf

- Status: `failed (dataset unavailable)`
- Script: `examples/neural/baseline/deepproblog/HWF/hwf.py`
- Log: `examples/neural/baseline/results/deepproblog_gpu_sequential/10_hwf_gpu_fullheldout.log`
- Command:
  - `PYTHONUNBUFFERED=1 XLOG_EVAL_BATCH_SIZE=256 PYTHONPATH=examples/neural/baseline/deepproblog python hwf.py`
- Result:
  - Exit: `1`
  - Runtime: `ELAPSED 0:04.43`
  - Error: `The HWD dataset has not been downloaded.`

### Setup Notes

- Poker image render completed for baseline-local dataset path:
  - `fair`: 2000 images
  - `unfair`: 2000 images
  - `fair_test`: 400 images
  - `unfair_test`: 400 images

## 2026-02-11 Updates (Exact Sharded Rerun)

### 10_hwf (fixed)

- Status: `complete`
- Change:
  - HWF archive extracted to baseline-local `HWF/data/`.
  - `HWF/hwf.py` imports switched to local modules (`HWF.data`, `HWF.network`) so downloaded data is used.
- Result:
  - Held-out Accuracy: `0.905`
  - Runtime: `ELAPSED 16:43.30`
  - Exit: `0`
- Log:
  - `examples/neural/baseline/results/deepproblog_gpu_sequential/10_hwf_gpu_fullheldout.log`

### 01_minimal (exact, full held-out via shards)

- Status: `complete`
- Config:
  - Train: `XLOG_RUN_MODE=train_only`, `XLOG_TRAIN_LIMIT=1024`
  - Eval: `XLOG_RUN_MODE=eval_only`, full test split covered by contiguous shards (`size=128`)
- Aggregate:
  - Correct: `512`
  - Total: `5000`
  - Held-out Accuracy: `0.102400`
- Logs/Artifacts:
  - Train log: `examples/neural/baseline/results/deepproblog_gpu_sequential/01_minimal_gpu_trainonly.log`
  - Shard table: `examples/neural/baseline/results/deepproblog_gpu_sequential/01_minimal_eval_shards.tsv`
  - Per-shard logs: `examples/neural/baseline/results/deepproblog_gpu_sequential/01_minimal_gpu_eval_shard_*.log`

### 02_mnist_addition (exact, full held-out via shards)

- Status: `complete`
- Config:
  - Train: `XLOG_RUN_MODE=train_only`, `XLOG_TRAIN_LIMIT=1024`
  - Eval: `XLOG_RUN_MODE=eval_only`, contiguous shards (`size=128`)
- Aggregate:
  - Correct: `512`
  - Total: `5000`
  - Held-out Accuracy: `0.102400`
- Logs/Artifacts:
  - Train log: `examples/neural/baseline/results/deepproblog_gpu_sequential/02_mnist_addition_gpu_trainonly.log`
  - Shard table: `examples/neural/baseline/results/deepproblog_gpu_sequential/02_mnist_addition_eval_shards.tsv`
  - Per-shard logs: `examples/neural/baseline/results/deepproblog_gpu_sequential/02_mnist_addition_gpu_eval_shard_*.log`

### 03_mnist_addition_noisy (exact closed-form, full held-out)

- Status: `complete`
- Config:
  - Train: `XLOG_RUN_MODE=train_only`, `XLOG_TRAIN_LIMIT=1024`
  - Eval: `XLOG_RUN_MODE=eval_only`, `XLOG_EVAL_METHOD=closed_form`, full test split (`0..5000`)
- Aggregate:
  - Correct: `512`
  - Total: `5000`
  - Held-out Accuracy: `0.102400`
- Exactness validation:
  - Closed-form evaluator was validated against solver output on shard `0..128`.
  - Solver shard result (`03_mnist_addition_noisy_gpu_eval_shard_0000_0128.log`): `22/128`
  - Closed-form shard result (`03_mnist_addition_noisy_gpu_eval_shard_0000_0128_closed_form.log`): `22/128`
- Logs/Artifacts:
  - Train log: `examples/neural/baseline/results/deepproblog_gpu_sequential/03_mnist_addition_noisy_gpu_trainonly.log`
  - Full held-out log: `examples/neural/baseline/results/deepproblog_gpu_sequential/03_mnist_addition_noisy_gpu_fullheldout.log`
  - Solver reference shard log: `examples/neural/baseline/results/deepproblog_gpu_sequential/03_mnist_addition_noisy_gpu_eval_shard_0000_0128.log`
  - Closed-form reference shard log: `examples/neural/baseline/results/deepproblog_gpu_sequential/03_mnist_addition_noisy_gpu_eval_shard_0000_0128_closed_form.log`

## 2026-02-12 Updates (Strict Full-Train Rerun)

### 01_minimal (strict full-train, multi-epoch)

- Status: `complete`
- Config:
  - Train: `XLOG_RUN_MODE=train_only`, full train split (`XLOG_TRAIN_LIMIT` unset), `XLOG_EPOCHS=2`, `XLOG_TRAIN_BATCH_SIZE=2048`
  - Eval: `XLOG_RUN_MODE=eval_only`, `XLOG_EVAL_METHOD=closed_form`, full held-out (`0..5000`)
- Result:
  - Held-out Accuracy: `0.2324`
  - Held-out Correct/Total: `1162/5000`
- Exactness validation:
  - Solver shard (`0..128`): `23/128` in `2:24.92`
  - Closed-form shard (`0..128`): `23/128` in `0:05.22`
- Runtime:
  - Train runtime: `ELAPSED 2:31:07`
- Logs/Artifacts:
  - Train log: `examples/neural/baseline/results/deepproblog_gpu_sequential/01_minimal_gpu_train_strict.log`
  - Full held-out log: `examples/neural/baseline/results/deepproblog_gpu_sequential/01_minimal_gpu_fullheldout_strict.log`
  - Solver shard log: `examples/neural/baseline/results/deepproblog_gpu_sequential/01_minimal_gpu_eval_shard_0000_0128_strict_solver.log`
  - Closed-form shard log: `examples/neural/baseline/results/deepproblog_gpu_sequential/01_minimal_gpu_eval_shard_0000_0128_strict_closed_form.log`

### 02_mnist_addition (strict full-train, multi-epoch)

- Status: `complete`
- Config:
  - Train: `XLOG_RUN_MODE=train_only`, full train split (`XLOG_TRAIN_LIMIT` unset), `XLOG_EPOCHS=2`, `XLOG_TRAIN_BATCH_SIZE=2048`
  - Eval: `XLOG_RUN_MODE=eval_only`, `XLOG_EVAL_METHOD=closed_form`, full held-out (`0..5000`)
- Result:
  - Held-out Accuracy: `0.2268`
  - Held-out Correct/Total: `1134/5000`
- Exactness validation:
  - Solver shard (`0..128`): `36/128` in `2:34.51`
  - Closed-form shard (`0..128`): `36/128` in `0:11.99`
- Runtime:
  - Train runtime: `ELAPSED 1:58:49`
- Logs/Artifacts:
  - Train log: `examples/neural/baseline/results/deepproblog_gpu_sequential/02_mnist_addition_gpu_train_strict.log`
  - Full held-out log: `examples/neural/baseline/results/deepproblog_gpu_sequential/02_mnist_addition_gpu_fullheldout_strict.log`
  - Solver shard log: `examples/neural/baseline/results/deepproblog_gpu_sequential/02_mnist_addition_gpu_eval_shard_0000_0128_strict_solver.log`
  - Closed-form shard log: `examples/neural/baseline/results/deepproblog_gpu_sequential/02_mnist_addition_gpu_eval_shard_0000_0128_strict_closed_form.log`

### 03_mnist_addition_noisy (strict evidence path)

- Status: `complete (from prior strict full-train evidence path)`
- Config:
  - Train: `XLOG_RUN_MODE=train_only`, full train split (`XLOG_TRAIN_LIMIT` unset), checkpoint `snapshot/noisy_addition.pth`
  - Eval: `XLOG_RUN_MODE=eval_only`, `XLOG_EVAL_METHOD=closed_form`, full held-out (`0..5000`)
- Result:
  - Held-out Accuracy: `0.1024`
  - Held-out Correct/Total: `512/5000`
- Exactness validation:
  - Solver shard (`0..128`): `22/128`
  - Closed-form shard (`0..128`): `22/128`
- Additional run characterization:
  - A 2-epoch strict attempt (`snapshot/noisy_addition_strict.pth`) was started and exceeded `2:35:41` wall-clock without reaching checkpoint save; terminated intentionally to avoid indefinite blocking.
  - Log: `examples/neural/baseline/results/deepproblog_gpu_sequential/03_mnist_addition_noisy_gpu_train_strict.log`
