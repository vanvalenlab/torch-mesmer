"""Parity tests: training-target transforms (``torch_mesmer.transforms``) vs
``deepcell-tf/deepcell/utils/transform_utils.py`` and
``deepcell-toolbox/deepcell_toolbox/utils.py``.

The port's pixelwise transform is channels-first (classes on axis 0), so the
reference is called with ``data_format='channels_first'``.
"""

import warnings

import numpy as np
import pytest

from torch_mesmer import transforms as tt
from torch_mesmer import utils as pu

from tests.conftest import random_blobs


@pytest.fixture(scope='module')
def labels_2d():
    return [random_blobs((96, 96), n_cells=20, seed=s) for s in range(4)]


@pytest.fixture(scope='module')
def labels_3d():
    return random_blobs((6, 40, 40), n_cells=8, radius=(3, 6), seed=0)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# erode_edges / fill_holes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('erosion_width', [0, 1, 2])
@pytest.mark.parametrize('port_fn', [tt.erode_edges, pu.erode_edges], ids=['transforms', 'utils'])
def test_erode_edges_matches_2d(ref, labels_2d, port_fn, erosion_width):
    for lab in labels_2d:
        np.testing.assert_array_equal(port_fn(lab, erosion_width),
                                      ref.utils.erode_edges(lab, erosion_width))


@pytest.mark.parametrize('port_fn', [tt.erode_edges, pu.erode_edges], ids=['transforms', 'utils'])
def test_erode_edges_matches_3d(ref, labels_3d, port_fn):
    np.testing.assert_array_equal(port_fn(labels_3d, 1), ref.utils.erode_edges(labels_3d, 1))


def test_fill_holes_matches(ref, labels_2d):
    rng = np.random.default_rng(0)
    lab = labels_2d[0].copy()
    # Punch small and large holes into the objects.
    for _ in range(40):
        r, c = rng.integers(2, 94, size=2)
        s = rng.integers(1, 4)
        lab[r:r + s, c:c + s] = 0
    for size in (5, 10, 15):
        np.testing.assert_array_equal(_quiet(pu.fill_holes, lab, size=size),
                                      _quiet(ref.utils.fill_holes, lab, size=size))


# ---------------------------------------------------------------------------
# pixelwise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dilation_radius', [None, 1, 2])
@pytest.mark.parametrize('separate_edge_classes', [False, True])
def test_pixelwise_transform_matches(ref, labels_2d, dilation_radius, separate_edge_classes):
    for lab in labels_2d:
        port = tt.pixelwise_transform(lab, dilation_radius=dilation_radius,
                                      separate_edge_classes=separate_edge_classes)
        refv = ref.transform_utils.pixelwise_transform(
            lab, dilation_radius=dilation_radius, data_format='channels_first',
            separate_edge_classes=separate_edge_classes)
        np.testing.assert_array_equal(port, refv)


@pytest.mark.parametrize('dilation_radius', [None, 1])
def test_pixelwise_transform_is_one_hot(labels_2d, dilation_radius):
    for lab in labels_2d:
        out = tt.pixelwise_transform(lab, dilation_radius=dilation_radius)
        assert out.shape == (3,) + lab.shape
        np.testing.assert_array_equal(out.sum(axis=0), 1)


def test_pixelwise_transform_matches_3d(ref, labels_3d):
    port = tt.pixelwise_transform(labels_3d, dilation_radius=1)
    refv = ref.transform_utils.pixelwise_transform(labels_3d, dilation_radius=1,
                                                   data_format='channels_first')
    np.testing.assert_array_equal(port, refv)


# ---------------------------------------------------------------------------
# inner distance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kwargs', [
    dict(alpha='auto'),
    dict(alpha='auto', erosion_width=1),
    dict(alpha=0.1),
    dict(alpha='auto', beta=2),
    dict(alpha='auto', bins=4),
])
def test_inner_distance_transform_matches(ref, labels_2d, kwargs):
    for lab in labels_2d:
        port = tt.inner_distance_transform_2d(lab, **kwargs)
        refv = _quiet(ref.transform_utils.inner_distance_transform_2d, lab, **kwargs)
        np.testing.assert_array_equal(port, refv)


def test_inner_distance_rejects_bad_alpha():
    with pytest.raises(ValueError):
        tt.inner_distance_transform_2d(random_blobs((32, 32), 3), alpha='bogus')


# ---------------------------------------------------------------------------
# outer distance
# ---------------------------------------------------------------------------

def test_outer_distance_transform_unnormalized_matches(ref, labels_2d):
    for lab in labels_2d:
        np.testing.assert_array_equal(tt.outer_distance_transform_2d(lab, normalize=False),
                                      ref.transform_utils.outer_distance_transform_2d(lab, normalize=False))


@pytest.mark.parametrize('kwargs', [dict(), dict(erosion_width=1), dict(bins=4)])
def test_outer_distance_transform_matches(ref, labels_2d, kwargs):
    for lab in labels_2d:
        np.testing.assert_array_equal(tt.outer_distance_transform_2d(lab, **kwargs),
                                      ref.transform_utils.outer_distance_transform_2d(lab, **kwargs))


# ---------------------------------------------------------------------------
# transform_masks: the batched wrapper the loader uses
# ---------------------------------------------------------------------------

def test_transform_masks_pixelwise_matches(ref, labels_2d):
    y = np.stack(labels_2d)[:, None]  # (B, 1, H, W)
    port = tt.transform_masks(y, 'pixelwise', dilation_radius=1)
    refv = np.stack([ref.transform_utils.pixelwise_transform(lab, dilation_radius=1,
                                                             data_format='channels_first')
                     for lab in labels_2d])
    np.testing.assert_array_equal(port, refv)


def test_transform_masks_inner_distance_matches(ref, labels_2d):
    y = np.stack(labels_2d)[:, None]
    port = tt.transform_masks(y, 'inner-distance', erosion_width=1, alpha='auto')
    refv = np.stack([_quiet(ref.transform_utils.inner_distance_transform_2d, lab,
                            erosion_width=1, alpha='auto') for lab in labels_2d])[:, None]
    np.testing.assert_array_equal(port, refv)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

@pytest.mark.data
def test_transforms_on_tissuenet(ref, tissuenet_test_slice):
    _, y = tissuenet_test_slice
    for img in y:
        for channel in img:
            lab = channel.astype(np.int32)
            np.testing.assert_array_equal(
                tt.pixelwise_transform(lab, dilation_radius=1),
                ref.transform_utils.pixelwise_transform(lab, dilation_radius=1, data_format='channels_first'))
            np.testing.assert_array_equal(
                tt.inner_distance_transform_2d(lab, erosion_width=1, alpha='auto'),
                _quiet(ref.transform_utils.inner_distance_transform_2d, lab, erosion_width=1, alpha='auto'))
