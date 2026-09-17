Model Evaluation
================

The ``torch_mesmer.eval`` module is used to evaluate the performance of
a given set of trained model weights using the TissueNet testing set.

## Requirements

Evaluating model weights requires:

1. The model you wish to evaluate (e.g. a `.pth` file).
2. The TissueNet dataset

## Running model evaluation

```bash
python -m torch_mesmer.eval \
    --model-path <path-to-.pth-file> \
    --device <PyTorch device identifier> \
    --data-path <path-to-tissuenet-dataset>
```

There are 3 optional arguments for the evaluation script:

- ``--model-path`` specifies the path to the particular model you'd like
  to evaluate. If unspecified, the evaluation script will search the default
  Deepcell models location (`$HOME/.deepcell/models`) for the latest version
  of pre-trained model weights. If not found, an error is raised in which case
  the path to the model must be set explicitly.
- ``--device`` specifies the device on which to run the model. Must be a valid
  [`torch.device` string][torch-device], e.g. `"cpu"`, `"cuda"`, `"mps"`, etc.
- ``--data-path`` specifies the path to the TissueNet dataset. If unspecified, the
  evaluation script will search the default Deepcell datasets location
  (`$HOME/.deepcell/datasets`).

See `python -m torch_mesmer --help` for more information

[torch-device]: https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-device

## Evaluation results

The evaluation script generates a `.csv` containing the evaluation metrics for all
images in the TissueNet test set.
This file is named `eval_results_<timestamp>.csv` where `<timestamp>` is the timestamp
of the completed evaluation run.
The file is created in the directory from which `python -m torch_mesmer.eval`
was called.
