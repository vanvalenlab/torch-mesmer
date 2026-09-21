from pathlib import Path
import torch

import pytest


_has_model = any(
    p.name.startswith("torch-mesmer") for p in (Path.home() / ".deepcell/models").iterdir()
)


_has_gpu = (torch.cuda.is_available() or torch.backends.mps.is_available())


requires_gpu = pytest.mark.skipif(
    not _has_gpu,
    reason="Acceleration hardware required for this test.",
)


requires_model = pytest.mark.skipif(
    not _has_model,
    reason="Pre-trained model weights required for this test.",
)
