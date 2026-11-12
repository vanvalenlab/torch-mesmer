"""Tests for ``torch_mesmer.loss.SemanticLoss`` against the published recipe.

The reference here is hand-written NumPy:

* ``ref_weighted_cce`` transcribes ``weighted_categorical_crossentropy`` from
  ``deepcell-tf/deepcell/losses.py:67-74`` (the deepcell-tf file imports
  TensorFlow, so it can't be loaded directly).
* ``ref_semantic_loss`` applies the paper's head weighting
  (``Mesmer_training_notebook.ipynb`` cell 10):
  ``0.01 × WCE`` on the pixelwise (3-class) heads, plain MSE on the
  inner-distance (1-channel) heads, all heads summed with weight 1, and each
  head's per-pixel loss averaged over batch and pixels (Keras's default
  reduction).

Three earlier mistakes are covered below: a double softmax, the 0.01 weight on the
wrong heads, and a missing ``+1`` in the class weights.
"""

import numpy as np
import pytest
import torch

from torch_mesmer.loss import SemanticLoss

KERAS_EPSILON = 1e-7   # K.epsilon()
CE_WEIGHT = 0.01       # recipe: 0.01 on the cross-entropy heads
HEADS = [1, 3, 1, 3]   # nuclear inner, nuclear pixelwise, cell inner, cell pixelwise


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def ref_weighted_cce(y_true, y_pred, n_classes):
    """Per-pixel weighted categorical cross-entropy on probabilities.

    Channels-first: ``y_true`` and ``y_pred`` are (B, C, H, W); returns (B, H, W).
    Line-by-line from ``deepcell-tf/deepcell/losses.py:67-74``.
    """
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    reduce_axis = (0, 2, 3)  # every axis except the class axis
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    y_pred = np.clip(y_pred, KERAS_EPSILON, 1.0 - KERAS_EPSILON)
    total_sum = y_true.sum()
    class_sum = y_true.sum(axis=reduce_axis, keepdims=True)
    class_weights = 1.0 / n_classes * total_sum / (class_sum + 1.0)
    return -(y_true * np.log(y_pred) * class_weights).sum(axis=1)


def ref_semantic_loss(y_pred, y_true, n_semantic_classes=HEADS):
    total = 0.0
    start = 0
    for n in n_semantic_classes:
        p, t = y_pred[:, start:start + n], y_true[:, start:start + n]
        if n > 1:
            total += CE_WEIGHT * ref_weighted_cce(t, p, n).mean()
        else:
            total += np.mean((p.astype(np.float64) - t) ** 2)
        start += n
    return total


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _one_hot_targets(rng, shape, n_classes=3, drop_class=None):
    """Spatially blocky one-hot targets (B, n_classes, H, W)."""
    B, H, W = shape
    coarse = rng.integers(0, n_classes, size=(B, H // 8, W // 8))
    classes = coarse.repeat(8, axis=1).repeat(8, axis=2)
    if drop_class is not None:
        classes[classes == drop_class] = (drop_class + 1) % n_classes
    return np.moveaxis(np.eye(n_classes)[classes], -1, 1).astype(np.float32)


def _softmax(logits, axis=1):
    e = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


@pytest.fixture
def batch():
    """A (y_pred, y_true) pair with the 8-channel Mesmer layout, as float32
    tensors. y_pred's pixelwise heads are softmax outputs, as the model emits."""
    rng = np.random.default_rng(0)
    shape = (2, 32, 32)
    preds, trues = [], []
    for n in HEADS:
        if n == 1:
            trues.append(rng.random((shape[0], 1, *shape[1:])).astype(np.float32))
            preds.append(rng.random((shape[0], 1, *shape[1:])).astype(np.float32))
        else:
            trues.append(_one_hot_targets(rng, shape))
            preds.append(_softmax(rng.normal(size=(shape[0], n, *shape[1:]))).astype(np.float32))
    return torch.from_numpy(np.concatenate(preds, 1)), torch.from_numpy(np.concatenate(trues, 1))


# ---------------------------------------------------------------------------
# The reference itself
# ---------------------------------------------------------------------------

def test_reference_perfect_prediction_is_near_zero():
    t = _one_hot_targets(np.random.default_rng(1), (2, 32, 32))
    assert ref_weighted_cce(t, t, 3).mean() < 1e-5


def test_reference_missing_class_is_finite():
    t = _one_hot_targets(np.random.default_rng(2), (2, 32, 32), drop_class=0)
    p = _softmax(np.random.default_rng(3).normal(size=t.shape))
    assert t[:, 0].sum() == 0
    assert np.isfinite(ref_weighted_cce(t, p, 3).mean())


# ---------------------------------------------------------------------------
# The port: structure that already matches the recipe
# ---------------------------------------------------------------------------

def test_heads_are_summed(batch):
    """The total is the plain sum of per-head losses (recipe: no loss_weights)."""
    y_pred, y_true = batch
    total = SemanticLoss(HEADS)(y_pred, y_true)
    parts, start = 0.0, 0
    for n in HEADS:
        parts = parts + SemanticLoss([n])(y_pred[:, start:start + n], y_true[:, start:start + n])
        start += n
    torch.testing.assert_close(total, parts)


def test_loss_is_finite_and_differentiable(batch):
    y_pred, y_true = batch
    y_pred = y_pred.clone().requires_grad_(True)
    loss = SemanticLoss(HEADS)(y_pred, y_true)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(y_pred.grad).all()
    assert y_pred.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Softmax applied once, 0.01 on the pixelwise heads, class weights with ``+1``
# ---------------------------------------------------------------------------

def test_ce_perfect_prediction_is_near_zero():
    t = torch.from_numpy(_one_hot_targets(np.random.default_rng(4), (2, 32, 32)))
    loss = SemanticLoss([3])(t.clone(), t)
    assert loss.item() < 1e-4


def test_mse_head_has_weight_one(batch):
    y_pred, y_true = batch
    loss = SemanticLoss([1])(y_pred[:, :1], y_true[:, :1])
    expected = torch.mean((y_pred[:, :1] - y_true[:, :1]) ** 2)
    torch.testing.assert_close(loss, expected)


def test_missing_class_gives_finite_loss():
    rng = np.random.default_rng(5)
    t = torch.from_numpy(_one_hot_targets(rng, (2, 32, 32), drop_class=0))
    p = torch.from_numpy(_softmax(rng.normal(size=tuple(t.shape))).astype(np.float32))
    assert t[:, 0].sum() == 0
    assert torch.isfinite(SemanticLoss([3])(p, t))


def test_semantic_loss_matches_recipe(batch):
    y_pred, y_true = batch
    port = SemanticLoss(HEADS)(y_pred, y_true).item()
    expected = ref_semantic_loss(y_pred.numpy(), y_true.numpy())
    assert port == pytest.approx(expected, rel=1e-5)


def test_ce_weight_scales_only_the_ce_heads(batch):
    y_pred, y_true = batch
    a = SemanticLoss(HEADS, ce_weight=0.01)(y_pred, y_true)
    b = SemanticLoss(HEADS, ce_weight=0.02)(y_pred, y_true)
    ce = SemanticLoss([3, 3], ce_weight=0.01)(
        torch.cat([y_pred[:, 1:4], y_pred[:, 5:8]], 1), torch.cat([y_true[:, 1:4], y_true[:, 5:8]], 1))
    torch.testing.assert_close(b - a, ce)


def test_ce_gradient_is_finite_for_saturated_predictions():
    """Probabilities of exactly 0 and 1 (a saturated softmax) are clamped, not log(0)."""
    t = torch.from_numpy(_one_hot_targets(np.random.default_rng(6), (2, 32, 32)))
    p = t.roll(1, dims=1).clone().requires_grad_(True)   # confidently wrong everywhere
    loss = SemanticLoss([3])(p, t)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(p.grad).all()
