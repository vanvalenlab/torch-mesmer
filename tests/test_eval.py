"""Tests for the scoring in ``torch_mesmer.eval``.

The eval scripts must report pooled (micro-averaged) recall, precision and F1,
as deepcell's ``Metrics.summarize_object_metrics_df`` and the Mesmer paper do.
"""

import warnings

import numpy as np
import pytest

from torch_mesmer import eval as ev

from tests.conftest import HAND_BUILT_CASES


def _evaluate(yp, yt):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ev.evaluate(yp, yt)


def test_evaluate_is_pooled(random_label_pairs):
    yt = np.stack([p[0] for p in random_label_pairs[:8]])
    yp = np.stack([p[1] for p in random_label_pairs[:8]])
    per_image, s = _evaluate(yp, yt)
    assert len(per_image) == 8
    correct, n_true, n_pred = (per_image[k].sum() for k in ('correct_detections', 'n_true', 'n_pred'))
    r, p = correct / n_true, correct / n_pred
    assert s['recall'] == round(r, 4)
    assert s['precision'] == round(p, 4)
    assert s['f1'] == round(2 * r * p / (r + p), 4)
    assert s['n_true'] == n_true and s['correct_detections'] == correct
    # The per-image means are reported separately.
    assert s['mean_per_image_recall'] == pytest.approx(per_image['recall'].mean())
    assert s['mean_per_image_f1'] == pytest.approx(per_image['f1'].mean())


def test_pooled_differs_from_mean_of_images():
    """One image with 1 cell (perfect), one with 9 cells (all missed): pooled
    recall is 1/10, while the mean of per-image recall is 0.5."""
    yt_a, _ = HAND_BUILT_CASES['perfect']
    big = np.zeros_like(yt_a)
    for k in range(9):
        r, c = divmod(k, 3)
        big[2 + 6 * r:6 + 6 * r, 2 + 6 * c:6 + 6 * c] = k + 1
    yt = np.stack([yt_a, big])
    yp = np.stack([yt_a, np.zeros_like(big)])
    _, s = _evaluate(yp, yt)
    assert s['recall'] == 0.1
    assert s['mean_per_image_recall'] == 0.5


def test_score_predictions_compartments(random_label_pairs):
    yt = np.stack([np.stack([p[0], p[1]]) for p in random_label_pairs[:4]])
    yp = np.stack([np.stack([p[1], p[1]]) for p in random_label_pairs[:4]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        per_image, summaries = ev.score_predictions(yp, yt, ['nuclear', 'whole-cell'])
    assert list(summaries.index) == ['nuclear', 'whole-cell']
    assert len(per_image) == 8
    assert set(per_image['compartment']) == {'nuclear', 'whole-cell'}
    # Channel 1 of the prediction equals channel 1 of the truth: perfect score.
    assert summaries.loc['whole-cell', 'f1'] == 1.0
    assert summaries.loc['nuclear', 'f1'] < 1.0
