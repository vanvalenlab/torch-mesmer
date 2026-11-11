"""Parity tests: the inference pipeline (``torch_mesmer.postprocess_utils``,
``torch_mesmer.utils``) vs deepcell-toolbox and the deepcell-tf Mesmer
application.

The port is channels-first ``(B, C, H, W)``; the reference is channels-last
``(B, H, W, C)``. Every comparison moves the channel axis explicitly.

Two differences from the reference are intentional or tolerated, and are tested as
such: the nuclear-first head order, and skimage versus cv2 for resizing
single-channel label images.
"""

import warnings

import numpy as np
import pytest
import scipy.ndimage as nd
from skimage.segmentation import find_boundaries

from torch_mesmer import postprocess_utils as pp
from torch_mesmer import transforms as tt
from torch_mesmer import utils as pu

from tests.conftest import random_blobs

MODEL_SHAPE = (256, 256)


def cl(x):
    """channels-first -> channels-last"""
    return np.moveaxis(x, 1, -1)


def cf(x):
    """channels-last -> channels-first"""
    return np.moveaxis(x, -1, 1)


def _image(shape, seed=0, blank=None):
    """Positive, skewed "fluorescence" image, channels-last (B, H, W, C).
    ``blank=(b, c)`` zeros one channel to exercise the constant-channel path."""
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 1.0, size=shape).astype(np.float32)
    x = nd.gaussian_filter(x, sigma=(0, 1.5, 1.5, 0))
    if blank is not None:
        x[blank[0], ..., blank[1]] = 0
    return x


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_percentile_threshold_matches(ref):
    x = _image((2, 96, 80, 2), blank=(1, 0))
    np.testing.assert_array_equal(cl(pu.percentile_threshold(cf(x))),
                                  ref.processing.percentile_threshold(x))


@pytest.mark.parametrize('kernel_size', [None, 64])
def test_histogram_normalization_matches(ref, kernel_size):
    x = _image((2, 96, 80, 2), blank=(0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        port = pu.histogram_normalization(cf(x), kernel_size=kernel_size)
        refv = ref.processing.histogram_normalization(x, kernel_size=kernel_size)
    np.testing.assert_array_equal(cl(port), refv)


@pytest.mark.parametrize('kwargs', [
    {},
    {'threshold': False},
    {'normalize': False},
    {'percentile': 99.0, 'kernel_size': 64},
])
def test_mesmer_preprocess_matches(ref, kwargs):
    x = _image((2, 160, 144, 2), seed=1, blank=(1, 1))
    port = pp.mesmer_preprocess(cf(x), **kwargs)
    refv = ref.app.mesmer_preprocess(x, **kwargs)
    np.testing.assert_array_equal(cl(port), refv)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

TILE_SHAPES = [
    (1, 600, 520, 3),
    (2, 256, 600, 2),   # one side exactly 256: spline blending is skipped (same in both)
    (1, 300, 300, 1),
    (1, 512, 512, 2),
]


@pytest.mark.parametrize('shape', TILE_SHAPES)
def test_tile_image_matches(ref, shape):
    x = np.random.default_rng(0).random(shape).astype(np.float32)
    tiles_p, info_p = pp.tile_image(cf(x), MODEL_SHAPE, stride_ratio=0.75)
    tiles_r, info_r = ref.utils.tile_image(x, MODEL_SHAPE, stride_ratio=0.75)

    np.testing.assert_array_equal(cl(tiles_p), tiles_r)
    assert set(info_p) == set(info_r)
    for key in info_r:
        if key == 'image_shape':  # the port stores it channels-first
            assert tuple(info_p[key]) == tuple(cf(np.empty(info_r[key])).shape)
        else:
            assert info_p[key] == info_r[key], key


@pytest.mark.parametrize('shape', TILE_SHAPES)
def test_untile_image_matches(ref, shape):
    x = np.random.default_rng(1).random(shape).astype(np.float32)
    tiles_p, info_p = pp.tile_image(cf(x), MODEL_SHAPE, stride_ratio=0.75)
    tiles_r, info_r = ref.utils.tile_image(x, MODEL_SHAPE, stride_ratio=0.75)

    out_p = pp.untile_image(tiles_p, info_p)
    out_r = ref.utils.untile_image(tiles_r, info_r)
    np.testing.assert_array_equal(cl(out_p), out_r)
    # The spline windows sum to 1, so tiling then untiling is (nearly) lossless.
    np.testing.assert_allclose(cl(out_p), x, atol=1e-3)


@pytest.mark.parametrize('shape', [
    (1, 200, 200, 2),   # smaller than the model: reflect-pad
    (1, 256, 200, 2),   # one side exactly 256, other smaller
    (1, 201, 150, 2),   # odd padding
    (2, 300, 300, 2),   # larger: tile
])
def test_tile_input_and_untile_output_match(ref, shape):
    x = np.random.default_rng(2).random(shape).astype(np.float32)
    tiles_p, info_p = pp.tile_input(cf(x), MODEL_SHAPE)
    tiles_r, info_r = ref.app.tile_input(x, MODEL_SHAPE)
    np.testing.assert_array_equal(cl(tiles_p), tiles_r)

    out_p = pp.untile_output(tiles_p, info_p)
    out_r = ref.app.untile_output(tiles_r, info_r, MODEL_SHAPE)
    np.testing.assert_array_equal(cl(out_p), out_r)
    assert out_p.shape == cf(x).shape


@pytest.mark.parametrize('shape', [(1, 2, 200, 600), (2, 2, 600, 200), (1, 2, 100, 300), (1, 2, 255, 257)])
def test_tile_input_one_side_small_one_side_large(shape):
    """Deliberate improvement over the reference (which
    pads the long side too and crashes): only the short side is padded, the
    long side is tiled, every tile is model-sized, and untiling gives back the input."""
    x = np.random.default_rng(3).random(shape).astype(np.float32)
    tiles, info = pp.tile_input(x, MODEL_SHAPE)
    assert tiles.shape[-2:] == MODEL_SHAPE
    np.testing.assert_allclose(pp.untile_output(tiles, info), x, atol=1e-3)


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('image_mpp', [0.25, 0.65, 0.75, 1.0])
def test_resize_input_matches(ref, image_mpp):
    x = np.random.default_rng(4).random((2, 2, 203, 190)).astype(np.float32)
    port = pp.resize_input(x, image_mpp, 0.5)
    refv = ref.app.resize_input(cl(x), image_mpp, 0.5)
    np.testing.assert_array_equal(cl(port), refv)


def test_resize_input_same_mpp_is_noop():
    x = np.random.default_rng(5).random((1, 2, 100, 90)).astype(np.float32)
    assert pp.resize_input(x, 0.5, 0.5) is x


def test_resize_input_none_mpp_is_noop():
    x = np.random.default_rng(5).random((1, 2, 100, 90)).astype(np.float32)
    assert pp.resize_input(x, None, 0.5) is x


def _label_stack(n_channels, shape=(101, 95)):
    return np.stack([random_blobs(shape, 20, seed=c) for c in range(n_channels)])[None].astype(np.int64)


@pytest.mark.parametrize('out_hw', [(150, 140), (400, 380), (406, 380)])
def test_resize_output_multichannel_matches(ref, out_hw):
    lab = _label_stack(2)
    port = pp.resize_output(lab, (1, 2, *out_hw))
    refv = ref.app.resize_output(cl(lab), (1, *out_hw, 2))
    np.testing.assert_array_equal(cl(port), refv)


@pytest.mark.parametrize('out_hw', [(150, 140), (400, 380), (406, 380)])
def test_resize_output_single_channel_tolerates_boundary_shifts(ref, out_hw):
    """Tolerated difference: for a single-channel label image the
    reference uses cv2 INTER_NEAREST while the port uses skimage order-0. The
    two libraries place pixel centres differently, so a few boundary pixels
    shift by one. Allow that, and nothing more: under 5% of pixels may differ,
    and only within 1 px of a label boundary."""
    lab = _label_stack(1)
    port = cl(pp.resize_output(lab, (1, 1, *out_hw)))[0, ..., 0]
    refv = ref.app.resize_output(cl(lab), (1, *out_hw, 1))[0, ..., 0]
    differs = port != refv
    assert differs.mean() < 0.05
    near_boundary = nd.binary_dilation(find_boundaries(refv, mode='thick'))
    assert not (differs & ~near_boundary).any()
    assert set(np.unique(port)) <= set(np.unique(lab))  # no new labels invented


# ---------------------------------------------------------------------------
# deep_watershed and mesmer_postprocess
# ---------------------------------------------------------------------------

def _fake_heads(labels, seed):
    """Plausible (inner_distance, pixelwise[3]) model outputs for one label
    image: smoothed targets plus a little noise."""
    rng = np.random.default_rng(seed)
    inner = tt.inner_distance_transform_2d(labels, alpha='auto')
    inner = nd.gaussian_filter(inner, 1) + 0.02 * rng.random(inner.shape)
    pix = tt.pixelwise_transform(labels).astype(np.float64)
    pix = nd.gaussian_filter(pix, sigma=(0, 1, 1)) + 0.02 * rng.random(pix.shape)
    pix /= pix.sum(axis=0, keepdims=True)
    return inner[None], pix


@pytest.fixture(scope='module')
def fake_model_output():
    """A port-layout model output, (B=2, 8, 128, 128), nuclear heads first:
    [nuc inner, nuc edge, nuc interior, nuc bg, cell inner, cell edge, cell interior, cell bg]."""
    batch = []
    for b in range(2):
        nuc = random_blobs((128, 128), 25, radius=(3, 6), seed=10 + b)
        cell = random_blobs((128, 128), 25, radius=(5, 10), seed=20 + b)
        n_inner, n_pix = _fake_heads(nuc, seed=b)
        c_inner, c_pix = _fake_heads(cell, seed=100 + b)
        batch.append(np.concatenate([n_inner, n_pix, c_inner, c_pix]))
    return np.stack(batch).astype(np.float32)


def _ref_heads(out8):
    """Port layout -> reference ``format_output_mesmer`` input: a channels-last
    list ``[cell inner, cell pixelwise, nuc inner, nuc pixelwise]`` (the port is nuclear first)."""
    x = cl(out8)
    return [x[..., 4:5], x[..., 5:8], x[..., 0:1], x[..., 1:4]]


PORT_NUC = dict(maxima_threshold=0.1, maxima_smooth=0, interior_threshold=0.075, interior_smooth=1,
                small_objects_threshold=15, fill_holes_threshold=15, radius=2, pixel_expansion=0)
PORT_CELL = dict(PORT_NUC, maxima_smooth=1)
REF_CELL = dict(maxima_threshold=0.075, maxima_smooth=0, interior_threshold=0.2, interior_smooth=2,
                small_objects_threshold=15, fill_holes_threshold=15, radius=2, pixel_expansion=None)

WATERSHED_KWARGS = [
    dict(pixel_expansion=None),
    dict(pixel_expansion=1),
    dict(pixel_expansion=2, maxima_smooth=1),
    dict(pixel_expansion=None, maxima_algorithm='peak_local_max'),
    dict(PORT_NUC, label_erosion=1),
    PORT_CELL,
    REF_CELL,
]


@pytest.mark.parametrize('kwargs', WATERSHED_KWARGS, ids=lambda k: ','.join(f'{a}={b}' for a, b in k.items()))
def test_deep_watershed_matches(ref, fake_model_output, kwargs):
    two = fake_model_output[:, [4, 6]]  # cell inner distance, cell interior
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        port = pp.deep_watershed(two, maxima_index=0, interior_index=1, **kwargs)
        refv = ref.deep_watershed.deep_watershed([cl(two[:, :1]), cl(two[:, 1:])],
                                                 maxima_index=0, interior_index=1, **kwargs)
    assert port.max() > 5  # the fixture really produces objects
    np.testing.assert_array_equal(cl(port), refv)


def test_deep_watershed_defaults_match(ref, fake_model_output):
    two = fake_model_output[:, [4, 6]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        port = pp.deep_watershed(two, maxima_index=0, interior_index=1)
        refv = ref.deep_watershed.deep_watershed([cl(two[:, :1]), cl(two[:, 1:])],
                                                 maxima_index=0, interior_index=1)
    np.testing.assert_array_equal(cl(port), refv)


def test_deep_watershed_rejects_invalid_algorithm(fake_model_output):
    with pytest.raises(ValueError):
        pp.deep_watershed(fake_model_output[:, [4, 6]], maxima_index=0, interior_index=1,
                          maxima_algorithm='not-an-algorithm')


def test_deep_watershed_warns_on_deprecated_kwargs(fake_model_output):
    # match= so that unrelated library DeprecationWarnings don't satisfy the check.
    with pytest.warns(DeprecationWarning, match='min_distance'):
        pp.deep_watershed(fake_model_output[:, [4, 6]], maxima_index=0, interior_index=1, min_distance=3)


@pytest.mark.parametrize('old, new, value', [
    ('min_distance', 'radius', 3),
    ('distance_threshold', 'interior_threshold', 0.3),
    ('detection_threshold', 'maxima_threshold', 0.2),
])
def test_deep_watershed_deprecated_alias_value_is_used(ref, fake_model_output, old, new, value):
    """As in the reference, the deprecated alias's value wins over the new name."""
    two = fake_model_output[:, [4, 6]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        via_alias = pp.deep_watershed(two, **{old: value})
        via_new = pp.deep_watershed(two, **{new: value})
        refv = ref.deep_watershed.deep_watershed([cl(two[:, :1]), cl(two[:, 1:])], **{old: value})
    np.testing.assert_array_equal(via_alias, via_new)
    np.testing.assert_array_equal(cl(via_alias), refv)


def test_deep_watershed_needs_indices_for_more_than_two_channels(fake_model_output):
    """The old default interior_index=-1 silently read the whole-cell background."""
    with pytest.raises(ValueError, match='maxima_index'):
        pp.deep_watershed(fake_model_output)


@pytest.mark.parametrize('kwargs', [dict(maxima_index=8, interior_index=6), dict(maxima_index=0, interior_index=-9)])
def test_deep_watershed_rejects_out_of_range_index(fake_model_output, kwargs):
    with pytest.raises(ValueError, match='out of range'):
        pp.deep_watershed(fake_model_output, **kwargs)


def test_deep_watershed_rejects_wrong_rank(fake_model_output):
    with pytest.raises(ValueError, match='channels-first'):
        pp.deep_watershed(fake_model_output[0], maxima_index=4, interior_index=6)


@pytest.mark.parametrize('compartment', ['whole-cell', 'nuclear', 'both'])
def test_mesmer_postprocess_matches(ref, fake_model_output, compartment):
    """With explicit head indices, the port and the reference give identical
    masks. For 'both' the port returns [nuclear, whole-cell] on axis 1 and the
    reference returns [whole-cell, nuclear] on the last axis (intentional)."""
    kw = dict(PORT_NUC, pixel_expansion=None)
    port = pp.mesmer_postprocess(
        fake_model_output, compartment=compartment,
        whole_cell_kwargs=dict(kw, maxima_index=4, interior_index=6),
        nuclear_kwargs=dict(kw, maxima_index=0, interior_index=2))
    refv = ref.app.mesmer_postprocess(
        ref.app.format_output_mesmer(_ref_heads(fake_model_output)), compartment=compartment,
        whole_cell_kwargs=kw, nuclear_kwargs=kw)
    if compartment == 'both':
        refv = refv[..., ::-1]
    np.testing.assert_array_equal(cl(port), refv)


@pytest.mark.parametrize('compartment, maxima_index, interior_index', [
    ('whole-cell', 4, 6),
    ('nuclear', 0, 2),
])
def test_mesmer_postprocess_picks_heads_by_default(fake_model_output, compartment, maxima_index, interior_index):
    """Head indices come from MESMER_HEAD_INDICES."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        default = pp.mesmer_postprocess(fake_model_output, compartment=compartment)
        explicit = pp.deep_watershed(fake_model_output, maxima_index=maxima_index,
                                     interior_index=interior_index)
    np.testing.assert_array_equal(default, explicit)


def test_mesmer_postprocess_both_is_nuclear_then_whole_cell(fake_model_output):
    """Intentional: 'both' stacks [nuclear, whole-cell] on axis 1."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        both = pp.mesmer_postprocess(fake_model_output, compartment='both')
        nuc = pp.mesmer_postprocess(fake_model_output, compartment='nuclear')
        cell = pp.mesmer_postprocess(fake_model_output, compartment='whole-cell')
    np.testing.assert_array_equal(both, np.concatenate([nuc, cell], axis=1))
    assert not np.array_equal(nuc, cell)
