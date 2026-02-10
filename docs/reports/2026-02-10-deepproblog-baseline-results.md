# DeepProbLog Baseline Execution Report (2026-02-10)

## Scope

- Installed `deepproblog==2.0.6`.
- Copied upstream DeepProbLog examples into `examples/neural/baseline/deepproblog`.
- Ran all neural baseline example entrypoints one-by-one.
- Captured logs in `examples/neural/baseline/results/deepproblog`.

## Environment

- Python: `3.10`
- Torch: `2.10.0+cu128`
- CUDA available: `True`
- DeepProbLog: `2.0.6`

## Dataset and Asset Preparation

- Synced full upstream examples tree from `/tmp/deepproblog-upstream-20260210/src/deepproblog/examples` into baseline folder.
- Synced same tree into installed package path (`~/.local/lib/python3.10/site-packages/deepproblog/examples`) so `deepproblog.examples.*` imports had full code/data assets.
- Generated Coins image datasets using Blender 2.79 wrapper and full label files:
  - installed package path: `.../deepproblog/examples/Coins/data/image_data/{train,test}`
- Generated Poker image datasets from full label files:
  - baseline path: `examples/neural/baseline/deepproblog/Poker/data/images/{fair,unfair,fair_test,unfair_test}`
  - installed package path: `.../deepproblog/examples/Poker/data/images/{fair,unfair,fair_test,unfair_test}`
- HWF download script executed, but upstream Google Drive source is not publicly retrievable anymore (details in results).

## Execution Policy

- Every script was executed directly from its example directory.
- Hard timeouts were used to prevent unbounded runs.
- Status file: `examples/neural/baseline/results/deepproblog/run_status_final.tsv`

## Final Status Matrix

| example | timeout_s | exit_code | status | elapsed_s |
|---|---:|---:|---|---:|
| `01_minimal` | 300 | 124 | timeout | 301 |
| `02_mnist_addition` | 300 | 124 | timeout | 301 |
| `03_mnist_addition_noisy` | 300 | 124 | timeout | 301 |
| `04_coins` | 600 | 124 | timeout | 601 |
| `05_poker` | 600 | 1 | error | 4 |
| `05_poker_rerun` | 600 | 124 | timeout | 601 |
| `06_clutrr_rerun` | 300 | 124 | timeout | 301 |
| `07_forth_add` | 600 | 0 | ok | 34 |
| `08_forth_sort` | 600 | 0 | ok | 12 |
| `09_forth_wap` | 600 | 124 | timeout | 601 |
| `10_hwf` | 300 | 1 | error | 4 |

## Key Observations by Example

### 01 minimal (`minimal/addition_minimal.py`)

- Reached iteration 800 within timeout.
- Loss improved from `2.7575` (iter 100) to `0.6656` (iter 800).
- Timed out before epoch completion.

### 02 MNIST addition (`MNIST/addition.py`)

- Reached iteration 800 within timeout.
- Loss improved from `2.7669` (iter 100) to `1.4381` (iter 800).
- Timed out before epoch completion and final confusion-matrix evaluation.

### 03 MNIST noisy addition (`MNIST/addition_noisy.py`)

- Reached iteration 700 within timeout.
- Loss improved from `2.8667` (iter 100) to `2.0286` (iter 700).
- Timed out before epoch completion.

### 04 Coins (`Coins/coins.py`)

- Reached test accuracy milestones before timeout:
  - `0.4` -> `0.5` -> `0.6` -> `1.0`
- Reached `Accuracy 1.0` at iteration 60.
- Timed out later during continued training/inference activity.

### 05 Poker (`Poker/poker.py`)

- First run failed immediately with missing local image files (`data/images/fair_test/0_0.png`).
- After generating baseline-local images and rerunning:
  - Initial test accuracy logged: `0.27`
  - Reached iteration 10 (`Average Loss: 0.0980`)
  - Timed out at 600s.

### 06 CLUTRR (`CLUTRR/clutrr.py`)

- Reached validation accuracy `0.99` and then `1.0` during training.
- During post-training test-dataset evaluation, encountered SWI-Prolog foreign return value error:
  - `domain_error(foreign_return_value, ...)` in `rel_extract_extern/2`
- Run ended under timeout code, with this error present in logs.

### 07 Forth Add (`Forth/Add/add.py`)

- Completed successfully (`exit 0`) in 34s.
- Dev accuracy progressed up to `0.9375`.

### 08 Forth Sort (`Forth/Sort/sort.py`)

- Completed successfully (`exit 0`) in 12s.
- Dev accuracy logged at `0.1875` and later `0.0625`.

### 09 Forth WAP (`Forth/WAP/wap.py`)

- Initial test accuracy: `0.085`.
- Reached iteration 10 (`Average Loss: 4.1417`).
- Timed out at 600s.

### 10 HWF (`HWF/hwf.py`)

- Failed immediately because HWF dataset could not be downloaded.
- `download_hwf.sh` target (`https://drive.google.com/uc?id=1G07kw-wK-rqbg_85tuB7FNfA49q8lvoy`) is not retrievable via `gdown` in current state.

## Artifacts

- Per-run logs: `examples/neural/baseline/results/deepproblog/*.log`
- Final status table: `examples/neural/baseline/results/deepproblog/run_status_final.tsv`

## Notes

- Generated runtime artifacts (downloaded MNIST files, rendered Poker images, and checkpoint outputs) were removed from the tracked baseline tree after execution to keep repository diffs source-focused and reproducible via documented commands.
