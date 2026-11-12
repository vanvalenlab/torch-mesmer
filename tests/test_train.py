"""Tests for the optimiser setup in ``torch_mesmer.train`` and the loader
settings in ``torch_mesmer.loaders`` against the published recipe
(the Mesmer paper's training notebook): Keras ``Adam(lr=1e-4, clipnorm=0.001)`` with
ε = 1e-7, ``lr · 0.99^epoch``, ``steps_per_epoch = n // batch_size`` and seed 0.

The clipping reference is ``tf.clip_by_norm`` (what Keras ``clipnorm`` applies
to every gradient), transcribed in NumPy: ``t · c / max(‖t‖₂, c)``.
"""

import numpy as np
import pytest
import torch
from torch import nn

from torch_mesmer.loaders import create_data_loaders
from torch_mesmer.loss import SemanticLoss
from torch_mesmer.train import (ADAM_EPSILON, CLIPNORM, clip_grad_norm_per_tensor_,
                                make_optimizer, train_step)


def ref_clip_by_norm(t, clip_norm):
    norm = np.sqrt((t.astype(np.float64) ** 2).sum())
    return t * clip_norm / max(norm, clip_norm)


def _model_with_grads(rng, scales):
    """A model whose parameters have gradients with norms of roughly ``scales``."""
    model = nn.Sequential(*[nn.Linear(4, 4, bias=False) for _ in scales])
    for p, s in zip(model.parameters(), scales):
        p.grad = torch.from_numpy(rng.normal(size=tuple(p.shape)).astype(np.float32))
        p.grad *= s / p.grad.norm()
    return model


# ---------------------------------------------------------------------------
# Per-tensor gradient clipping
# ---------------------------------------------------------------------------

def test_clip_matches_keras_clip_by_norm():
    rng = np.random.default_rng(0)
    model = _model_with_grads(rng, [1e-5, 5e-4, 1e-3, 0.02, 3.0])
    before = [p.grad.numpy().copy() for p in model.parameters()]
    norms = clip_grad_norm_per_tensor_(model.named_parameters(), max_norm=CLIPNORM)
    for g0, p, n in zip(before, model.parameters(), norms):
        np.testing.assert_allclose(p.grad.numpy(), ref_clip_by_norm(g0, CLIPNORM), rtol=1e-6)
        assert n.item() == pytest.approx(np.linalg.norm(g0), rel=1e-5)


def test_clip_is_per_tensor_not_global():
    """A large gradient on one tensor must not shrink a small one on another."""
    rng = np.random.default_rng(1)
    model = _model_with_grads(rng, [1e-4, 10.0])
    small_before = next(model.parameters()).grad.clone()
    clip_grad_norm_per_tensor_(model.named_parameters())
    small, large = (p.grad for p in model.parameters())
    torch.testing.assert_close(small, small_before)
    assert large.norm().item() == pytest.approx(CLIPNORM, rel=1e-5)


def test_clip_leaves_zero_gradient_alone():
    model = nn.Linear(3, 3)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    clip_grad_norm_per_tensor_(model.named_parameters())
    assert all(torch.isfinite(p.grad).all() and (p.grad == 0).all() for p in model.parameters())


def test_clip_skips_parameters_without_gradients():
    model = nn.Linear(3, 3)
    model.weight.grad = torch.ones_like(model.weight)
    norms = clip_grad_norm_per_tensor_(model.named_parameters())
    assert len(norms) == 1
    assert model.bias.grad is None


def test_clip_names_the_non_finite_parameter():
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    model[1].bias.grad[0] = float('nan')
    with pytest.raises(FloatingPointError, match=r'1 of 4 .*1\.bias'):
        clip_grad_norm_per_tensor_(model.named_parameters())


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


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------

class _TinyHeads(nn.Module):
    """Outputs the [1, 3] head layout: one ReLU channel, one 3-class softmax."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 1)

    def forward(self, x):
        y = self.conv(x)
        return torch.cat([torch.relu(y[:, :1]), torch.softmax(y[:, 1:], dim=1)], dim=1)


def _tiny_batch(rng):
    x = torch.from_numpy(rng.normal(size=(2, 2, 16, 16)).astype(np.float32))
    classes = rng.integers(0, 3, size=(2, 16, 16))
    y = np.concatenate([rng.random((2, 1, 16, 16)), np.moveaxis(np.eye(3)[classes], -1, 1)], 1)
    return x, torch.from_numpy(y.astype(np.float32))


def test_train_step_reduces_loss():
    torch.manual_seed(0)
    model = _TinyHeads()
    x, y = _tiny_batch(np.random.default_rng(2))
    loss_fn = SemanticLoss([1, 3])
    optimizer, _ = make_optimizer(model, lr=1e-2)
    first = train_step(model, loss_fn, optimizer, x, y)
    for _ in range(50):
        last = train_step(model, loss_fn, optimizer, x, y)
    assert not first.requires_grad
    assert last < first


def test_train_step_return_heads_matches_total():
    """With ``return_heads=True`` the step gives the same total loss, plus the per-head parts."""
    x, y = _tiny_batch(np.random.default_rng(4))
    results = []
    for return_heads in (False, True):
        torch.manual_seed(0)
        model = _TinyHeads()
        optimizer, _ = make_optimizer(model)
        results.append(train_step(model, SemanticLoss([1, 3]), optimizer, x, y, return_heads=return_heads))
    total, (total_h, heads) = results
    assert heads.shape == (2,) and not heads.requires_grad
    torch.testing.assert_close(total_h, total)
    torch.testing.assert_close(heads.sum(), total)


def test_train_step_rejects_non_finite_loss():
    model = _TinyHeads()
    x, y = _tiny_batch(np.random.default_rng(3))
    x[0, 0, 0, 0] = float('nan')
    optimizer, _ = make_optimizer(model)
    with pytest.raises(FloatingPointError, match='Training loss is nan'):
        train_step(model, SemanticLoss([1, 3]), optimizer, x, y)


# ---------------------------------------------------------------------------
# Loader: drop_last and seeding
# ---------------------------------------------------------------------------

def _fake_split(n):
    # No 'meta': the loader does not read the pixel size.
    return {'X': np.zeros((n, 2, 8, 8), np.float32), 'y': np.zeros((n, 2, 8, 8), np.int32)}


def test_loaders_drop_the_partial_batch():
    train, val = create_data_loaders(_fake_split(21), _fake_split(13), batch_size=8, num_workers=0,
                                     crop_size=8, min_objects=0)
    assert train.drop_last and val.drop_last
    assert len(train) == 21 // 8
    assert len(val) == 13 // 8


def test_train_shuffle_is_seeded():
    def order(seed):
        train, _ = create_data_loaders(_fake_split(40), None, batch_size=8, num_workers=0, seed=seed,
                                       crop_size=8, min_objects=0)
        return [list(b) for b in train.batch_sampler]

    assert order(0) == order(0)
    assert order(0) != order(1)
