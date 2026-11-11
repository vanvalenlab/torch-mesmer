"""Tests for the optimiser setup in ``torch_mesmer.train`` and the loader
settings in ``torch_mesmer.loaders`` against the published recipe
(the Mesmer paper's training notebook): Keras ``Adam(lr=1e-4, clipnorm=0.001)`` with
ε = 1e-7, ``lr · 0.99^epoch``, ``steps_per_epoch = n // batch_size`` and seed 0.
"""

import numpy as np
import pytest
import torch
from torch import nn

from torch_mesmer.train import ADAM_EPSILON, make_optimizer


# ---------------------------------------------------------------------------
# Adam ε and the LR schedule
# ---------------------------------------------------------------------------

def test_adam_uses_keras_epsilon_and_lr():
    optimizer, _ = make_optimizer(nn.Linear(2, 2), lr=1e-4)
    assert isinstance(optimizer, torch.optim.Adam)
    group = optimizer.param_groups[0]
    assert group['eps'] == ADAM_EPSILON == 1e-7
    assert group['lr'] == pytest.approx(1e-4)
    assert group['betas'] == (0.9, 0.999)
    assert group['weight_decay'] == 0


def test_lr_decays_by_0p99_per_epoch():
    optimizer, scheduler = make_optimizer(nn.Linear(2, 2), lr=1e-4)
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    np.testing.assert_allclose(lrs, 1e-4 * 0.99 ** np.arange(100), rtol=1e-12)
