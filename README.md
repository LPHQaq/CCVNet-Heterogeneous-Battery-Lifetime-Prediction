# CCVNet

Code release for **CCVNet**: a **context-aware capacity-voltage network** for battery lifetime prediction from early-cycle capacity-voltage difference (CVD) curves and cell-level metadata/descriptors.

This repository is organized as a paper reproduction repo:

- training, preprocessing, and cache generation live in Python modules;
- figure generation lives in Jupyter notebooks under `Visualization/`;
- experiment settings are defined by YAML config files under `configs/`.

## Repository Layout

```text
CCVNet/
|-- configs/
|   |-- default.yaml
|   |-- major/
|   |-- ablation/
|   `-- all/
|-- data/
|   |-- preprocess/
|   `-- processed/
|-- results/
|-- scripts/
|   |-- run_preprocess.py
|   |-- run_train.py
|   `-- run_evaluate.py
|-- src/
|   `-- ccvnet/
|       |-- ablation/
|       |-- experiments/
|       `-- models/
|-- tests/
`-- Visualization/
    |-- homogeneous dataset analysis.ipynb
    |-- main_baseline.ipynb
    |-- small_data.ipynb
    |-- small_cycle.ipynb
    |-- transfer.ipynb
    |-- transfer_diagnose.ipynb
    |-- moe.ipynb
    `-- Graph/
```

## Installation

```bash
conda create -n ccvnet python=3.10
conda activate ccvnet
pip install -r requirements.txt
pip install -e .
```

For GPU training, install a PyTorch build that matches your CUDA environment first, then install the remaining requirements.

## Data Preprocessing

Dataset-specific preprocessing scripts are collected under `data/preprocess/`. The unified entry point is:

```bash
python scripts/run_preprocess.py --dataset all
```

You can also preprocess a single dataset:

```bash
python scripts/run_preprocess.py --dataset MATR
python scripts/run_preprocess.py --dataset HUST
```

By default, preprocessing writes CVD curve artifacts into `data/processed/`. Descriptor summary CSV files can also be exported as preprocessing artifacts, but training descriptors are rebuilt from the saved CVD files during training.

## Training Entry Points

The main training entry point is:

```bash
python scripts/run_train.py --experiment <name>
```

Available publish-facing experiment groups are:

### Major experiments

```bash
python scripts/run_train.py --experiment main_baseline
python scripts/run_train.py --experiment small_data
python scripts/run_train.py --experiment small_cycle
python scripts/run_train.py --experiment transfer
```

These default to:

- `configs/major/main_baseline.yaml`
- `configs/major/small_data.yaml`
- `configs/major/small_cycle.yaml`
- `configs/major/transfer.yaml`

### Ablation experiments

```bash
python scripts/run_train.py --experiment ablation-main_baseline
python scripts/run_train.py --experiment ablation-small_data
python scripts/run_train.py --experiment ablation-small_cycle
python scripts/run_train.py --experiment ablation-transfer
python scripts/run_train.py --experiment moe_baseline
python scripts/run_train.py --experiment moe_small_data
python scripts/run_train.py --experiment moe_early_cycle
python scripts/run_train.py --experiment descriptor_attribution
```

### Explicit config usage

```bash
python scripts/run_train.py --experiment major-main_baseline --config configs/major/main_baseline.yaml
```

## Config Structure

`configs/default.yaml` stores shared defaults such as:

- repository-local data and result paths;
- split policy and repeated seeds;
- default training hyperparameters;
- shared CVD input settings.

Each paper result block then has its own config file in `configs/major/`, `configs/ablation/`, or `configs/all/`.

## Results and Visualization

Training outputs are written to `results/` using repo-local cache directories that are consumed directly by the notebooks in `Visualization/`.

Main notebooks:

- `Visualization/main_baseline.ipynb`
- `Visualization/small_data.ipynb`
- `Visualization/small_cycle.ipynb`
- `Visualization/transfer.ipynb`
- `Visualization/transfer_diagnose.ipynb`
- `Visualization/moe.ipynb`
- `Visualization/homogeneous dataset analysis.ipynb`

Generated figure exports are written to `Visualization/Graph/`.

## Notes on Cached Results

This release is set up to work with repo-local cached result tables. Most benchmark-style training entry points support incremental cache reuse and can skip models whose result frames are already present.

The `transfer` pipeline is more heavyweight than the baseline benchmarks and may take substantially longer to rerun from scratch.

## Git Tracking

The repository ignores generated artifacts such as:

- `results/`
- `data/processed/`
- `Visualization/Graph/*.tiff`

This keeps the Git history focused on source code, configs, notebooks, and documentation.

## Citation

Citation metadata is included in `CITATION.cff`.
