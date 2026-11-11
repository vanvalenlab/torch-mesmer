"""Parity tests: ``torch_mesmer.metrics`` vs ``deepcell_toolbox.metrics``.

Every object count and score is expected to match the reference exactly. The
tests for the cases where the port still differs from the reference are added
together with the fixes.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from skimage.segmentation import relabel_sequential

from torch_mesmer import metrics as pm
from torch_mesmer import utils as pu

from tests.conftest import HAND_BUILT_CASES

# Every count reported by ObjectMetrics.to_dict.
COUNT_KEYS = [
    'n_true', 'n_pred', 'correct_detections', 'missed_detections', 'gained_detections',
    'split', 'merge', 'catastrophe', 'gained_det_from_split', 'missed_det_from_merge',
    'true_det_in_catastrophe', 'pred_det_in_catastrophe',
]
# The error counts (everything except correct_detections).
ERROR_KEYS = [k for k in COUNT_KEYS if k != 'correct_detections']
# Scores derived from correct_detections.
SCORE_KEYS = ['recall', 'precision', 'f1']
# Pixel-level scores (independent of the object matching).
PIXEL_KEYS = ['jaccard', 'dice']


def _object_metrics(module, y_true, y_pred, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return module.ObjectMetrics(y_true, y_pred, **kwargs).to_dict()


def _diff(port, ref, keys):
    """``{key: (port, ref)}`` for every key where the two disagree."""
    out = {}
    for k in keys:
        a, b = port[k], ref[k]
        both_nan = isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)
        if not both_nan and not np.isclose(a, b, rtol=0, atol=1e-12):
            out[k] = (a, b)
    return out


def _diff_over_pairs(ref, pairs, keys, **kwargs):
    """Compare port and reference on each pair; return {pair index: diff}."""
    bad = {}
    for i, (yt, yp) in enumerate(pairs):
        d = _diff(_object_metrics(pm, yt, yp, **kwargs),
                  _object_metrics(ref.metrics, yt, yp, **kwargs), keys)
        if d:
            bad[i] = d
    return bad


# ---------------------------------------------------------------------------
# Building blocks: verified identical
# ---------------------------------------------------------------------------

def test_compute_overlap_matches_cython_2d(ref):
    rng = np.random.default_rng(0)

    def boxes(n):
        lo = rng.integers(0, 60, size=(n, 2))
        hi = lo + rng.integers(0, 25, size=(n, 2))
        return np.hstack([lo, hi]).astype(np.float64)

    a, b = boxes(40), boxes(30)
    np.testing.assert_array_equal(pu.compute_overlap_vectorized(a, b), ref.compute_overlap(a, b))


def test_get_box_labels_matches(ref, random_label_pairs):
    yt, _ = random_label_pairs[0]
    boxes_p, labels_p = pu.get_box_labels(yt)
    boxes_r, labels_r = ref.metrics.get_box_labels(yt)
    np.testing.assert_array_equal(boxes_p, boxes_r)
    assert labels_p == labels_r


def test_match_nodes_matches(ref, random_label_pairs):
    # match_nodes takes a stack of frames: (frames, x, y).
    yt = np.stack([p[0] for p in random_label_pairs[:3]])
    yp = np.stack([p[1] for p in random_label_pairs[:3]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        np.testing.assert_array_equal(pu.match_nodes(yt, yp), ref.metrics.match_nodes(yt, yp))


def test_split_stack_matches(ref):
    arr = np.arange(2 * 64 * 48 * 3).reshape(2, 64, 48, 3)
    args = (True, 4, 1, 3, 2)
    np.testing.assert_array_equal(pu.split_stack(arr, *args), ref.metrics.split_stack(arr, *args))


# ---------------------------------------------------------------------------
# PixelMetrics
# ---------------------------------------------------------------------------

PIXEL_PROPS = ['recall', 'precision', 'f1', 'dice', 'jaccard']


def _pixel_props(module, yt, yp):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = module.PixelMetrics(yt, yp)
        return {k: getattr(m, k) for k in PIXEL_PROPS}


def _pixel_cases(random_label_pairs):
    yt, yp = random_label_pairs[2]
    empty = np.zeros_like(yt)
    return {'random': (yt, yp), 'both_empty': (empty, empty), 'true_empty': (empty, yp),
            'pred_empty': (yt, empty), 'identical': (yt, yt)}


def test_pixel_metrics_properties_match(ref, random_label_pairs):
    for name, (yt, yp) in _pixel_cases(random_label_pairs).items():
        port, refv = _pixel_props(pm, yt, yp), _pixel_props(ref.metrics, yt, yp)
        assert not _diff(port, refv, PIXEL_PROPS), name


# ---------------------------------------------------------------------------
# ObjectMetrics, 2D
# ---------------------------------------------------------------------------

def test_object_error_types_match_with_force_event_links(ref, random_label_pairs):
    """Error counts with force_event_links=True (the large-cell reduction is on)."""
    bad = _diff_over_pairs(ref, random_label_pairs, ERROR_KEYS + PIXEL_KEYS, force_event_links=True)
    assert not bad


def test_object_correct_and_scores_match_with_force_event_links(ref, random_label_pairs):
    bad = _diff_over_pairs(ref, random_label_pairs, ['correct_detections'] + SCORE_KEYS,
                           force_event_links=True)
    assert not bad


@pytest.mark.parametrize('name, expected', [
    ('perfect', dict(correct_detections=1)),
    ('split', dict(correct_detections=0, split=1)),
    ('merge', dict(correct_detections=0, merge=1)),
    # deepcell counts a catastrophe as a correct detection as well (is_correct
    # only excludes splits and merges), so correct_detections is 1 here.
    ('catastrophe', dict(correct_detections=1, catastrophe=1)),
    ('missed', dict(correct_detections=1, missed_detections=1)),
    ('gained', dict(correct_detections=1, gained_detections=1)),
    ('small_cell_in_merge', dict(correct_detections=0, merge=1, missed_detections=0)),
])
def test_reference_hand_built_expectations(ref, name, expected):
    """Sanity check of the reference itself on the hand-built cases, so the
    expectations the port is held to are the intended ones."""
    yt, yp = HAND_BUILT_CASES[name]
    refv = _object_metrics(ref.metrics, yt, yp)
    assert {k: refv[k] for k in expected} == expected


def test_clean_split_is_not_correct():
    yt, yp = HAND_BUILT_CASES['split']
    d = _object_metrics(pm, yt, yp)
    assert d['correct_detections'] == 0
    assert d['f1'] == 0


def test_clean_merge_is_not_correct():
    yt, yp = HAND_BUILT_CASES['merge']
    d = _object_metrics(pm, yt, yp)
    assert d['correct_detections'] == 0


def test_small_cell_in_merge_matches_with_force_event_links(ref):
    yt, yp = HAND_BUILT_CASES['small_cell_in_merge']
    port = _object_metrics(pm, yt, yp, force_event_links=True)
    refv = _object_metrics(ref.metrics, yt, yp, force_event_links=True)
    assert not _diff(port, refv, ERROR_KEYS)


def test_empty_frames_match(ref, empty_frame_case):
    name, yt, yp = empty_frame_case
    port = _object_metrics(pm, yt, yp)
    refv = _object_metrics(ref.metrics, yt, yp)
    assert not _diff(port, refv, COUNT_KEYS + SCORE_KEYS + PIXEL_KEYS), name


# ---------------------------------------------------------------------------
# Metrics (batch wrapper)
# ---------------------------------------------------------------------------

def _batch(random_label_pairs, n=8):
    yt = np.stack([p[0] for p in random_label_pairs[:n]])
    yp = np.stack([p[1] for p in random_label_pairs[:n]])
    return yt, yp


def _summary(module, yt, yp, **kwargs):
    m = module.Metrics('test', **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = m.calc_object_stats(yt, yp, progbar=False)
    return df, m.summarize_object_metrics_df(df)


SUMMARY_ERROR_KEYS = ['n_true', 'n_pred', 'jaccard', 'total_errors', 'gained_det_from_split',
                      'missed_det_from_merge', 'true_det_in_catastrophe', 'pred_det_in_catastrophe',
                      'gained_detections', 'missed_detections', 'split', 'merge', 'catastrophe']


def test_metrics_summary_error_counts_match(ref, random_label_pairs, capsys):
    yt, yp = _batch(random_label_pairs)
    _, port = _summary(pm, yt, yp, force_event_links=True)
    _, refv = _summary(ref.metrics, yt, yp, force_event_links=True)
    assert not _diff(port, refv, SUMMARY_ERROR_KEYS)


def test_metrics_summary_scores_match(ref, random_label_pairs, capsys):
    yt, yp = _batch(random_label_pairs)
    _, port = _summary(pm, yt, yp, force_event_links=True)
    _, refv = _summary(ref.metrics, yt, yp, force_event_links=True)
    assert not _diff(port, refv, ['correct_detections', 'recall', 'precision'])


def test_metrics_summary_is_pooled(random_label_pairs):
    """``summarize_object_metrics_df`` pools counts over images (micro average),
    as the reference does."""
    yt, yp = _batch(random_label_pairs)
    df, summary = _summary(pm, yt, yp)
    assert summary['recall'] == round(df['correct_detections'].sum() / df['n_true'].sum(), 4)
    assert summary['precision'] == round(df['correct_detections'].sum() / df['n_pred'].sum(), 4)


def test_metrics_relabels_non_sequential_input(random_label_pairs):
    """Non-sequential labels are relabelled before matching, in both."""
    yt, yp = _batch(random_label_pairs, n=2)
    yt_gappy = np.where(yt > 0, yt * 3 + 7, 0)
    df_seq, _ = _summary(pm, yt, yp)
    df_gappy, _ = _summary(pm, yt_gappy, yp)
    pd.testing.assert_frame_equal(df_seq, df_gappy)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

@pytest.mark.data
def test_object_metrics_on_tissuenet(ref, tissuenet_test_slice):
    """Ground truth vs. an eroded copy of itself, on real TissueNet labels."""
    from torch_mesmer.utils import erode_edges
    _, y = tissuenet_test_slice
    for img in y:
        for channel in img:
            yt, _, _ = relabel_sequential(channel.astype(np.int32))
            yp, _, _ = relabel_sequential(erode_edges(yt, 2))
            port = _object_metrics(pm, yt, yp, force_event_links=True)
            refv = _object_metrics(ref.metrics, yt, yp, force_event_links=True)
            assert not _diff(port, refv, ERROR_KEYS + PIXEL_KEYS)
