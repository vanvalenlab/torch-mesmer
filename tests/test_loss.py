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

The tests for each fix to the loss arrive with that fix.
"""

import numpy as np
import torch

from torch_mesmer.loss import SemanticLoss

KERAS_EPSILON = 1e-7   # K.epsilon()
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
# Softmax applied once, 0.01 on the pixelwise heads, class weights with ``+1``
# ---------------------------------------------------------------------------

def test_ce_perfect_prediction_is_near_zero():
    t = torch.from_numpy(_one_hot_targets(np.random.default_rng(4), (2, 32, 32)))
    loss = SemanticLoss([3])(t.clone(), t)
    assert loss.item() < 1e-4


def test_missing_class_gives_finite_loss():
    rng = np.random.default_rng(5)
    t = torch.from_numpy(_one_hot_targets(rng, (2, 32, 32), drop_class=0))
    p = torch.from_numpy(_softmax(rng.normal(size=tuple(t.shape))).astype(np.float32))
    assert t[:, 0].sum() == 0
    assert torch.isfinite(SemanticLoss([3])(p, t))


def test_ce_gradient_is_finite_for_saturated_predictions():
    """Probabilities of exactly 0 and 1 (a saturated softmax) are clamped, not log(0)."""
    t = torch.from_numpy(_one_hot_targets(np.random.default_rng(6), (2, 32, 32)))
    p = t.roll(1, dims=1).clone().requires_grad_(True)   # confidently wrong everywhere
    loss = SemanticLoss([3])(p, t)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(p.grad).all()
