"""Shared fixtures for the torch_mesmer test suite.

Synthetic label images are generated at fixed seeds so every run sees the same
inputs. ``tissuenet_test_slice`` loads a few real images and is skipped when
``~/.deepcell/tissuenet_v1-1/test.zarr`` is missing or empty.
"""

import os

# This runs on a shared server: cap thread pools before numpy/torch are imported.
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.ndimage as nd  # noqa: E402
from skimage.segmentation import relabel_sequential, watershed  # noqa: E402

from tests import _reference  # noqa: E402

TISSUENET_TEST_ZARR = Path.home() / '.deepcell' / 'tissuenet_v1-1' / 'test.zarr'


def pytest_configure(config):
    """Limit torch to ``OMP_NUM_THREADS`` threads for the whole test session."""
    import torch
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))


@pytest.fixture(autouse=True)
def _cap_torch_threads():
    """Put the thread cap back before every test, in case a test or import changed it.

    (``torch_mesmer.mesmer`` used to call ``torch.set_num_threads(24)`` at import
    time, which overrode the cap.)
    """
    import torch
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    yield


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def ref():
    """The deepcell reference modules (see ``tests/_reference.py``)."""
    try:
        return _reference.load()
    except _reference.ReferenceUnavailable as e:
        pytest.skip(str(e))


# ---------------------------------------------------------------------------
# Synthetic label images
# ---------------------------------------------------------------------------

def random_blobs(shape=(128, 128), n_cells=25, radius=(4, 10), seed=0):
    """Random touching "cells": seeds at random positions, each grown to a
    random radius, with touching cells divided by a watershed. Works in 2D and
    3D. Labels are sequential starting at 1."""
    rng = np.random.default_rng(seed)
    shape = tuple(shape)
    points = np.stack([rng.integers(0, s, size=n_cells) for s in shape], axis=1)
    markers = np.zeros(shape, dtype=np.int32)
    markers[tuple(points.T)] = np.arange(1, n_cells + 1)

    dist, nearest = nd.distance_transform_edt(markers == 0, return_indices=True)
    nearest_label = markers[tuple(nearest)]
    radii = np.concatenate([[0], rng.uniform(*radius, size=n_cells)])
    foreground = dist < radii[nearest_label]

    labels = watershed(dist, markers, mask=foreground)
    labels, _, _ = relabel_sequential(labels)
    return labels.astype(np.int32)


def perturb_labels(y, seed=0, p_drop=0.08, p_split=0.10, p_merge=0.10,
                   p_erode=0.15, n_gained=3):
    """Make a plausible "prediction" from a ground-truth label image, with
    misses, splits, merges, boundary jitter and false positives."""
    from skimage.segmentation import find_boundaries
    rng = np.random.default_rng(seed)
    pred = y.copy()
    next_label = int(y.max()) + 1

    for lab in range(1, int(y.max()) + 1):
        mask = y == lab
        if not mask.any():
            continue
        r = rng.random()
        if r < p_drop:
            pred[mask] = 0
        elif r < p_drop + p_split:
            # Half the splits cut through the middle; the rest chip off a small
            # sliver, which is the case the small-cell IoU boost exists for.
            coords = np.argwhere(mask)
            direction = rng.normal(size=y.ndim)
            proj = (coords - coords.mean(axis=0)) @ direction
            cut = 0.0 if rng.random() < 0.5 else np.quantile(proj, 0.95)
            side = proj > cut
            pred[tuple(coords[side].T)] = next_label
            next_label += 1
        elif r < p_drop + p_split + p_merge:
            ring = nd.binary_dilation(mask) & ~mask & (y > 0)
            neighbours = np.unique(y[ring])
            if neighbours.size:
                pred[y == neighbours[0]] = lab
        elif r < p_drop + p_split + p_merge + p_erode:
            pred[mask & find_boundaries(y, mode='inner')] = 0

    # False positives in the background.
    background = np.argwhere(y == 0)
    for _ in range(n_gained):
        if not len(background):
            break
        c = background[rng.integers(len(background))]
        grid = np.indices(y.shape)
        blob = sum((g - ci) ** 2 for g, ci in zip(grid, c)) <= rng.uniform(2, 4) ** 2
        pred[blob & (pred == 0)] = next_label
        next_label += 1

    pred, _, _ = relabel_sequential(pred)
    return pred.astype(np.int32)


N_RANDOM_PAIRS = 40


@pytest.fixture(scope='session')
def random_label_pairs():
    """40 (y_true, y_pred) pairs of 128x128 label images with mixed errors."""
    pairs = []
    for seed in range(N_RANDOM_PAIRS):
        y_true = random_blobs((128, 128), n_cells=30, seed=seed)
        y_pred = perturb_labels(y_true, seed=1000 + seed)
        pairs.append((y_true, y_pred))
    return pairs


@pytest.fixture(scope='session')
def random_label_volumes():
    """Five (y_true, y_pred) pairs of 3D label volumes (z, x, y)."""
    pairs = []
    for seed in range(5):
        y_true = random_blobs((8, 48, 48), n_cells=10, radius=(3, 7), seed=seed)
        y_pred = perturb_labels(y_true, seed=2000 + seed, n_gained=1)
        pairs.append((y_true, y_pred))
    return pairs


def _square(img, label, r0, r1, c0, c1):
    """Paint rows ``r0:r1`` and columns ``c0:c1`` of ``img`` with ``label``, in place."""
    img[r0:r1, c0:c1] = label


def _hand_built_cases():
    """Small hand-built cases with a known error type each."""
    cases = {}

    yt = np.zeros((40, 40), np.int32)
    _square(yt, 1, 5, 25, 5, 25)
    cases['perfect'] = (yt, yt.copy())

    yp = np.zeros_like(yt)
    _square(yp, 1, 5, 15, 5, 25)
    _square(yp, 2, 15, 25, 5, 25)
    cases['split'] = (yt, yp)

    yt = np.zeros((40, 40), np.int32)
    _square(yt, 1, 5, 15, 5, 25)
    _square(yt, 2, 15, 25, 5, 25)
    yp = np.zeros_like(yt)
    _square(yp, 1, 5, 25, 5, 25)
    cases['merge'] = (yt, yp)

    # Two true cells split vertically, two predicted cells split horizontally.
    yt = np.zeros((40, 40), np.int32)
    _square(yt, 1, 5, 25, 5, 15)
    _square(yt, 2, 5, 25, 15, 25)
    yp = np.zeros_like(yt)
    _square(yp, 1, 5, 15, 5, 25)
    _square(yp, 2, 15, 25, 5, 25)
    cases['catastrophe'] = (yt, yp)

    yt = np.zeros((40, 40), np.int32)
    _square(yt, 1, 5, 15, 5, 15)
    _square(yt, 2, 25, 35, 25, 35)
    yp = np.zeros_like(yt)
    _square(yp, 1, 5, 15, 5, 15)
    cases['missed'] = (yt, yp)

    yt = np.zeros((40, 40), np.int32)
    _square(yt, 1, 5, 15, 5, 15)
    yp = yt.copy()
    _square(yp, 2, 25, 35, 25, 35)
    cases['gained'] = (yt, yp)

    # Two 200 px true cells and one 4 px true cell, all merged into one
    # predicted cell. The 4 px cell's IoU with the merged
    # prediction (4/440) is below cutoff2, so without the small-cell boost it has
    # no graph edge and is counted as missed instead of as part of the merge.
    yt = np.zeros((40, 40), np.int32)
    _square(yt, 1, 5, 15, 5, 25)
    _square(yt, 2, 15, 25, 5, 25)
    _square(yt, 3, 25, 27, 10, 12)
    yp = np.zeros_like(yt)
    _square(yp, 1, 5, 27, 5, 25)
    cases['small_cell_in_merge'] = (yt, yp)

    return cases


HAND_BUILT_CASES = _hand_built_cases()


@pytest.fixture(params=sorted(HAND_BUILT_CASES))
def hand_built_case(request):
    """(name, y_true, y_pred) for each hand-built case."""
    yt, yp = HAND_BUILT_CASES[request.param]
    return request.param, yt, yp


@pytest.fixture(params=['both_empty', 'true_empty', 'pred_empty'])
def empty_frame_case(request):
    """(name, y_true, y_pred) where at least one frame has no objects."""
    yt = random_blobs((64, 64), n_cells=8, seed=7)
    empty = np.zeros_like(yt)
    return request.param, *{
        'both_empty': (empty, empty.copy()),
        'true_empty': (empty, yt),
        'pred_empty': (yt, empty),
    }[request.param]


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def tissuenet_test_slice():
    """The first 4 images of the TissueNet v1.1 test split, as
    ``(X, y)`` float32 arrays of shape (4, 2, 256, 256), nuclear channel first.

    Built by ``torch_mesmer/preprocess.py``. Skipped if the file is missing or
    incomplete.
    """
    if not TISSUENET_TEST_ZARR.exists():
        pytest.skip(f'{TISSUENET_TEST_ZARR} not found')
    import zarr
    try:
        store = zarr.open(str(TISSUENET_TEST_ZARR), mode='r')
        X = np.asarray(store['X'][:4])
        y = np.asarray(store['y'][:4])
    except (KeyError, FileNotFoundError, ValueError) as e:
        pytest.skip(f'{TISSUENET_TEST_ZARR} is incomplete: {e!r}')
    return X, y
