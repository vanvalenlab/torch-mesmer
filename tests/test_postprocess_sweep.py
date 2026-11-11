"""Tests for ``torch_mesmer.postprocess_sweep`` .

The sweep reuses work between settings, so it must still give exactly the
masks ``Mesmer.predict`` would: ``deep_watershed`` then ``resize_output``.
"""

import warnings

import numpy as np
import pytest
import scipy.ndimage as nd

from torch_mesmer import postprocess_sweep as ps
from torch_mesmer.metrics import Metrics
from torch_mesmer.postprocess_utils import deep_watershed, resize_output


def _synthetic_maps(shape, n_cells=25, seed=0):
    """Smooth blob maps resembling an inner-distance map and an interior map."""
    rng = np.random.default_rng(seed)
    seeds = np.zeros(shape)
    pts = rng.integers(8, np.array(shape) - 8, size=(n_cells, 2))
    seeds[pts[:, 0], pts[:, 1]] = rng.uniform(0.5, 1.0, n_cells)
    maxima = nd.gaussian_filter(seeds, 3)
    maxima = maxima / maxima.max() + rng.normal(0, 0.02, shape)
    interior = nd.gaussian_filter((nd.gaussian_filter(seeds, 6) > 0.002).astype(float), 1.5)
    interior = np.clip(interior + rng.normal(0, 0.05, shape), 0, 1)
    return maxima.astype(np.float32), interior.astype(np.float32)


SETTINGS = [
    {**ps.BASE_KWARGS},
    {**ps.BASE_KWARGS, 'maxima_threshold': 0.05, 'interior_threshold': 0.1, 'maxima_smooth': 1},
    {**ps.BASE_KWARGS, 'interior_smooth': 0, 'radius': 3, 'small_objects_threshold': 0},
    {**ps.BASE_KWARGS, 'pixel_expansion': 1, 'fill_holes_threshold': 0, 'small_objects_threshold': 40},
    {**ps.BASE_KWARGS, 'maxima_threshold': 0.05, 'interior_threshold': 0.1, 'maxima_smooth': 1,
     'small_objects_threshold': 30},  # shares its watershed with setting 1
]


@pytest.mark.parametrize('model_hw, original_hw', [
    ((256, 256), (256, 256)),   # 0.5 µm/px: no rescale
    ((512, 512), (256, 256)),   # 1.0 µm/px: model output is shrunk back
    ((187, 187), (256, 256)),   # 0.365 µm/px: model output is enlarged back
])
def test_segment_settings_matches_deep_watershed(model_hw, original_hw):
    maxima, interior = _synthetic_maps(model_hw, seed=model_hw[0])
    transforms = np.stack([maxima, interior])[None]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = ps.segment_settings(maxima, interior, SETTINGS, original_hw)
        for s, mask in zip(SETTINGS, got):
            ref = deep_watershed(transforms, maxima_index=0, interior_index=1, **s)
            ref = resize_output(ref, (1, 1) + original_hw).astype(int)[0, 0]
            assert mask.shape == original_hw
            assert ref.max() > 3  # the synthetic maps give real objects
            np.testing.assert_array_equal(mask, ref)


def test_pool_counts_matches_metrics_summary(random_label_pairs):
    pairs = random_label_pairs[:6]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        counts = np.array([ps.score_mask(yt, yp) for yt, yp in pairs])
        m = Metrics('t')
        df = m.calc_object_stats(np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs]),
                                 progbar=False)
    ref = m.summarize_object_metrics_df(df)
    got = ps.pool_counts(counts)
    for k in ('recall', 'precision', 'f1'):
        assert round(got[k], 4) == ref[k]
    for k in ps.COUNT_FIELDS:
        assert got[k] == ref[k]
    assert got['mean_per_image_f1'] == pytest.approx(df['f1'].mean())


def test_expand_grid_fills_base_and_merges_duplicates():
    entries = [
        {'compartment': 'nuclear', 'label': 'grid', 'grid': {'interior_threshold': [0.1, 0.2]}},
        {'compartment': 'nuclear', 'label': 'ref', 'grid': {'interior_threshold': [0.2]}},
        {'compartment': 'whole-cell', 'label': 'ref', 'grid': {'interior_threshold': [0.2]}},
    ]
    out = ps.expand_grid(entries)
    assert len(out) == 3
    assert out[1][1] == ['grid', 'ref']  # same nuclear setting, both labels kept
    assert all(set(kw) == set(ps.SWEEP_KEYS) for _, _, kw in out)
    with pytest.raises(ValueError):
        ps.expand_grid([{'compartment': 'nuclear', 'label': 'x', 'grid': {'bogus': [1]}}])
