Model Training
==============

The ``torch_mesmer.train`` module is used to train the model from the
TissueNet training dataset.

## Requirements

Training Mesmer requires:

1. The TissueNet dataset, or another labeled dataset in TissueNet format.
2. Acceleration hardware (e.g. GPU) with at least 32 GB of memory available

## Training configuration

The configuration options for the model itself and the training run are stored
[within the `train.py` script][train-config-src].
Altering training/model parameters requires editing `training.py` directly.

[train-config-src][https://github.com/vanvalenlab/torch-mesmer/blob/4e8082b04701e5b5a5b4d131d5eacda6621ab093/torch_mesmer/train.py#L147-L170]

## Start a training run

```bash
python -m torch_mesmer.train
```

## Training results

By default, the training products are stored in a `data/` directory which is
created in the working directory from which `torch_mesmer.train` is called.
The output directory has the following structure:

```
data/
├── logs
└── model
```

By default, the model that results from training can be found at
`data/model/<timestamp>/saved_model_best_dict.pth`

### Example: evaluating trained model results

A trained model can be evaluated using the `torch_mesmer.eval` module like so:

```bash
python -m torch_mesmer.eval --model-path data/model/<timestamp>/saved_model_best_dict.pth
```
