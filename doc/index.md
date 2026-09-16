``torch-mesmer`` documentation
==============================

```{toctree}
---
maxdepth: 1
hidden: true
---
tutorial
model_access
```

Welcome to the ``torch-mesmer`` documentation!

``torch-mesmer`` is a PyTorch implementation of the [Mesmer pipeline][mesmer-paper] for segmenting
multiplexed tissue images.
The pipeline is specifically designed for use with whole-slide tissue images.

[mesmer-paper]: https://www.nature.com/articles/s41587-021-01094-0

```{note}
This package is a PyTorch reimplementation of the [Mesmer pipeline][mesmer-tf] from the
TensorFlow-based [`deepcell-tf`][deepcell-tf] package.

[`deepcell-tf`][deepcell-tf] is no longer maintained - users are encouraged to use this package instead.
```

[deepcell-tf]: https://deepcell.readthedocs.io/en/master/#
[mesmer-tf]: https://deepcell.readthedocs.io/en/latest/app-gallery/mesmer.html

## Installation

The development version can be installed with:

```bash
pip install git+https://github.com/vanvalenlab/torch-mesmer.git
```

## Basic Usage

The `Mesmer` class provides the primary interface to the whole slide cell segmentation
pipeline.
The basic incantation:

```python
from torch_mesmer.mesmer import Mesmer

app = Mesmer()
mask = app.predict(...)
```

See the {doc}`tutorial` for further details.
