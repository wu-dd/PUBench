# PUBench

Positive-Unlabeled (PU) learning benchmark code. This repository contains runnable code for training, sweeping, and collecting results for the paper.

## Requirements

- Python 3.10
- PyTorch, torchvision
- NumPy, PIL, pandas, scikit-learn
- (Optional for figures) matplotlib, seaborn

## Data

- **CIFAR10**: Set `--data_dir` to the directory containing the CIFAR10 dataset (e.g. `/path/to/CIFAR10/`).
- **IMAGENETTE**: Set `--data_dir` to the Imagenette data root.
- **Letter / USPS**: Set `--data_dir` to the directory containing the Letter or USPS data (as used in the paper).
- **Creditcard**: Place `creditcard.csv` in `data/Creditcard/` (or set `--data_dir` to that directory). 

## Running

### Single run (train)

```bash
python -m train --data_dir /path/to/data/ --dataset CIFAR10 --algorithm uPU --hparams_seed 0 --trial_seed 0 --seed 0 --output_dir ./results/tmp/run1 --holdout_fraction 0.1 --skip_model_save --setting set1_1 --calibration False
```

### Sweep (multiple algorithms / settings / trials)

```bash
python sweep.py launch --data_dir=./data/Letter/ --command_launcher multi_gpu --n_hparams_from 0 --n_hparams 1 --n_trials_from 0 --n_trials 3 --datasets Letter --algorithms uPU nnPU nnPU_GA VPU Dist_PU PUSB --setting set1_1 set2_1 --output_dir=./results/tmp --skip_model_save --steps 20000
```

Use `--command_launcher local` for a single machine. Results are written under `output_dir` in per-run subdirectories.

### Collect results

After runs complete, aggregate results (and optionally output LaTeX):

```bash
python collect_results.py --input_dir ./results/tmp
# With LaTeX output:
python collect_results.py --input_dir ./results/tmp --latex
```

Redirect the latter to a file if you need a `.tex` summary.

## Settings

- `--setting`: Predefined PU settings. `set1_1`–`set5_1` (and `set1_2`–`set5_2`) use one-sample style; `set6_1`–`set10_1` (and `set6_2`–`set10_2`) use two-sample style. See `train.py` and `lib/misc.py` for the exact definitions.

## Structure

- `train.py`: Single training run.
- `sweep.py`: Launch many jobs (algorithms × settings × trials).
- `collect_results.py`: Aggregate run results from `output_dir`.
- `core/`: Algorithms and hyperparameter registry.
- `data/`: Dataset loaders and transforms.
- `lib/`: Utilities and reporting.

