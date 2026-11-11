"""Tests for the training data pipeline: ``torch_mesmer.augmentation`` and
``torch_mesmer.loaders``.
"""

import random

import numpy as np
import pytest
import scipy.ndimage as nd
import skimage.morphology
import torch
import torch.nn.functional as F

from torch_mesmer.augmentation import MultiTransform
from torch_mesmer.loaders import SegmentationDataset
from torch_mesmer.transforms import pixelwise_transform
from tests.conftest import random_blobs


def _fixed_transform(shape, crop_h, crop_w, top=0, left=0, angle=0.0, hflip=False, vflip=False, crop_size=256):
    t = MultiTransform(shape, dataset_type='train', crop_size=crop_size)
    t.crop_h, t.crop_w, t.top, t.left = crop_h, crop_w, top, left
    t.angle, t.do_hflip, t.do_vflip = angle, hflip, vflip
    return t


# ---------------------------------------------------------------------------
# MultiTransform
# ---------------------------------------------------------------------------

def test_val_is_identity_at_crop_size():
    x = torch.rand(3, 256, 256)
    t = MultiTransform((256, 256), dataset_type='val')
    torch.testing.assert_close(t(x, ['bilinear', 'nearest', 'nearest']), x, rtol=0, atol=0)


def test_val_centre_crops_larger_images():
    x = torch.rand(1, 300, 280)
    out = MultiTransform((300, 280), dataset_type='val')(x, ['bilinear'])
    torch.testing.assert_close(out, x[:, 22:278, 12:268], rtol=0, atol=0)


def test_zoom_only_is_a_plain_resize_without_antialiasing():
    # Angle 0, no flips: the single resampling must equal crop + bilinear
    # resize with align_corners=False and no antialiasing.
    x = torch.rand(2, 512, 512)
    t = _fixed_transform((512, 512), crop_h=330, crop_w=190, top=40, left=100)
    out = t(x, ['bilinear', 'bilinear'])
    ref = F.interpolate(x[None, :, 40:370, 100:290], size=(256, 256), mode='bilinear',
                        align_corners=False, antialias=False)[0]
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-4)


def test_zoom_only_nearest_matches_nearest_exact():
    # Labels use nearest sampling on the same coordinate grid as the image
    # (NEAREST_EXACT, no legacy floor offset).
    x = torch.randint(0, 50, (1, 512, 512)).float()
    t = _fixed_transform((512, 512), crop_h=300, crop_w=200, top=10, left=20)
    out = t(x, ['nearest'])
    ref = F.interpolate(x[None, :, 10:310, 20:220], size=(256, 256), mode='nearest-exact')[0]
    assert (out != ref).float().mean() < 0.01  # differences only at exact rounding ties


@pytest.mark.parametrize('mode', ['bilinear', 'nearest'])
def test_rotation_is_counter_clockwise(mode):
    x = torch.rand(1, 256, 256)
    t = _fixed_transform((256, 256), crop_h=256, crop_w=256, angle=90.0)
    torch.testing.assert_close(t(x, [mode]), torch.rot90(x, 1, dims=(1, 2)), rtol=0, atol=1e-4)


def test_flips():
    x = torch.rand(1, 256, 256)
    out = _fixed_transform((256, 256), 256, 256, hflip=True, vflip=True)(x, ['nearest'])
    torch.testing.assert_close(out, x.flip(-1).flip(-2), rtol=0, atol=0)


def test_rotation_corners_use_nearest_fill():
    # Corners exposed by rotation repeat the edge, they are not zero.
    x = torch.ones(2, 256, 256)
    out = _fixed_transform((256, 256), 256, 256, angle=45.0)(x, ['bilinear', 'nearest'])
    torch.testing.assert_close(out, x, rtol=0, atol=1e-6)


def test_crop_larger_than_image_is_edge_padded():
    x = torch.rand(1, 256, 256)
    t = _fixed_transform((256, 256), crop_h=300, crop_w=300, top=-20, left=-30)
    out = t(x, ['nearest'])
    assert out.shape == (1, 256, 256)
    assert torch.isfinite(out).all() and (out > 0).all()


def test_zoom_is_anisotropic_and_in_range():
    random.seed(0)
    hw = np.array([[t.crop_h, t.crop_w] for t in
                   (MultiTransform((512, 512), 'train', zoom_min=0.7) for _ in range(2000))])
    lo, hi = 0.7 * 256, 256 / 0.7
    assert hw.min() >= np.floor(lo) and hw.max() <= np.ceil(hi)
    assert hw.min() < lo + 3 and hw.max() > hi - 3
    assert (hw[:, 0] != hw[:, 1]).mean() > 0.9
    assert abs(np.corrcoef(hw.T)[0, 1]) < 0.1


def test_crop_position_uses_the_image_shape():
    # The crop offset is drawn over the real image, not a hard-coded 512 frame.
    random.seed(0)
    for _ in range(200):
        t = MultiTransform((400, 1000), 'train')
        assert 0 <= t.top <= 400 - t.crop_h and 0 <= t.left <= 1000 - t.crop_w


def test_rejects_bad_modes_and_shapes():
    t = MultiTransform((64, 64), 'train', crop_size=32)
    with pytest.raises(ValueError, match='unknown interpolation'):
        t(torch.rand(1, 64, 64), ['bicubic'])
    with pytest.raises(ValueError, match='interpolation modes'):
        t(torch.rand(2, 64, 64), ['bilinear'])
    with pytest.raises(ValueError, match='drawn for'):
        t(torch.rand(1, 32, 64), ['bilinear'])


# ---------------------------------------------------------------------------
# SegmentationDataset
# ---------------------------------------------------------------------------

def _sample(seed=0, shape=(512, 512)):
    """Nuclear and whole-cell labels, plus a 2-channel image that is bright
    exactly on each compartment's pixelwise *interior* class, so the image and
    that target channel should overlap after any augmentation."""
    whole = random_blobs(shape, n_cells=120, radius=(8, 16), seed=seed)
    nuclear = np.where(nd.binary_erosion(whole > 0, skimage.morphology.disk(3)), whole, 0)
    interiors = [pixelwise_transform(lab, dilation_radius=1)[1] for lab in (nuclear, whole)]
    rng = np.random.default_rng(seed)
    X = np.stack(interiors).astype(np.float32) * 100 + rng.random((2,) + shape, np.float32)
    y = np.stack([nuclear, whole]).astype(np.float32)
    return X[None], y[None]


def _iou(a, b):
    return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


@pytest.mark.parametrize('seed', range(4))
def test_augmented_sample_is_one_hot_and_aligned(seed):
    X, y = _sample()
    ds = SegmentationDataset(X, y, dataset_type='train', semantic_heads=[1, 3, 1, 3])
    random.seed(seed)
    x_out, y_out = ds[0]
    x_out, y_out = x_out.numpy(), y_out.numpy()

    assert x_out.shape == (2, 256, 256) and y_out.shape == (8, 256, 256)
    for pix in (y_out[1:4], y_out[5:8]):
        assert set(np.unique(pix)) <= {0.0, 1.0}
        np.testing.assert_array_equal(pix.sum(axis=0), 1)            # one-hot
        np.testing.assert_array_equal(pix[2], 1 - (pix[0] + pix[1]))  # bg = 1 - (edge + interior)

    # Image and labels moved together: the bright image regions are the
    # interior class. Aligned samples give about 0.985; shifting the labels by
    # a single pixel already drops it to 0.85-0.91.
    for c, interior in ((0, y_out[2]), (1, y_out[6])):
        assert _iou(x_out[c] > 0.5, interior > 0) > 0.95
    # The inner distance lives inside the cells.
    assert (y_out[4][y_out[7] == 1] < 0.05).mean() > 0.97


def test_augmentation_is_seeded():
    X, y = _sample()
    ds = SegmentationDataset(X, y, dataset_type='train')
    random.seed(3)
    a = ds[0]
    random.seed(3)
    b = ds[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_val_sample_is_not_augmented():
    X, y = _sample(shape=(256, 256))
    ds = SegmentationDataset(X, y, dataset_type='val')
    x_out, y_out = ds[0]
    np.testing.assert_allclose(x_out.numpy(), ds._normalize(X[0]), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(y_out.numpy(), ds._transform_labels(y[0]))
