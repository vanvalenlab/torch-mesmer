from pathlib import Path
import torch

import pytest


_model_path = Path.home() / ".deepcell/models"


_has_model = _model_path.exists() and any(
    p.name.startswith("torch-mesmer") for p in (_model_path).iterdir()
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
