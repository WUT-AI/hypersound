# HyperSound

## Setup

Setup conda environment:

```console
conda env create -f environment.yml
```

Populate `.env` file with settings from `.env.example`, e.g.:

```txt
DATA_DIR=~/datasets
RESULTS_DIR=~/results
WANDB_ENTITY=hypersound
WANDB_PROJECT=hypersound
```

Make sure that `pytorch-yard` is using the appropriate version (defined in `train.py`). If not, then correct package version with something like:

```console
pip install --force-reinstall pytorch-yard==2022.9.1
```

## Experiments

Default experiment:

```console
python train.py
```

Custom settings:

```console
python train.py cfg.learning_rate=0.01 cfg.pl.max_epochs=100
```

Isolated training of a target network on a single recording:

```console
python train_inr.py
```
