import pytest

import numpy as np

from .mask_utils import _transform_masks

def test_no_transform():
    num_classes = np.random.randint(5, size=1)[0]
    num_classes = max(1, num_classes)
    # test 2D masks
    mask = np.random.randint(num_classes, size=(5, 30, 30, 1))
    mask_transform = _transform_masks(
        mask, transform=None, data_format='channels_last')
    np.testing.assert_equal(mask_transform.shape, (5, 30, 30, num_classes))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    mask = np.random.randint(num_classes, size=(5, 1, 30, 30))
    mask_transform = _transform_masks(
        mask, transform=None, data_format='channels_first')
    np.testing.assert_equal(mask_transform.shape, (5, num_classes, 30, 30))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    # test 3D masks
    mask = np.random.randint(num_classes, size=(5, 10, 30, 30, 1))
    mask_transform = _transform_masks(
        mask, transform=None, data_format='channels_last')
    np.testing.assert_equal(mask_transform.shape, (5, 10, 30, 30, num_classes))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    mask = np.random.randint(num_classes, size=(5, 1, 10, 30, 30))
    mask_transform = _transform_masks(
        mask, transform=None, data_format='channels_first')
    np.testing.assert_equal(mask_transform.shape, (5, num_classes, 10, 30, 30))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

def test_fgbg_transform():
    num_classes = 2  # always 2 for fg and bg
    # test 2D masks
    mask = np.random.randint(3, size=(5, 30, 30, 1))
    mask_transform = _transform_masks(
        mask, transform='fgbg', data_format='channels_last')
    np.testing.assert_equal(mask_transform.shape, (5, 30, 30, num_classes))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    mask = np.random.randint(3, size=(5, 1, 30, 30))
    mask_transform = _transform_masks(
        mask, transform='fgbg', data_format='channels_first')
    np.testing.assert_equal(mask_transform.shape, (5, num_classes, 30, 30))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    # test 3D masks
    mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
    mask_transform = _transform_masks(
        mask, transform='fgbg', data_format='channels_last')
    np.testing.assert_equal(mask_transform.shape, (5, 10, 30, 30, num_classes))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    mask = np.random.randint(3, size=(5, 1, 10, 30, 30))
    mask_transform = _transform_masks(
        mask, transform='fgbg', data_format='channels_first')
    np.testing.assert_equal(mask_transform.shape, (5, num_classes, 10, 30, 30))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

def test_pixelwise_transform():
    # test 2D masks
    mask = np.random.randint(3, size=(5, 30, 30, 1))
    mask_transform = _transform_masks(
        mask, transform='pixelwise', data_format='channels_last',
        separate_edge_classes=True)
    np.testing.assert_equal(mask_transform.shape, (5, 30, 30, 4))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    mask = np.random.randint(3, size=(5, 1, 30, 30))
    mask_transform = _transform_masks(
        mask, transform='pixelwise', data_format='channels_first',
        separate_edge_classes=False)
    np.testing.assert_equal(mask_transform.shape, (5, 3, 30, 30))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    # test 3D masks
    mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
    mask_transform = _transform_masks(
        mask, transform='pixelwise', data_format='channels_last',
        separate_edge_classes=False)
    np.testing.assert_equal(mask_transform.shape, (5, 10, 30, 30, 3))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    mask = np.random.randint(3, size=(5, 1, 10, 30, 30))
    mask_transform = _transform_masks(
        mask, transform='pixelwise', data_format='channels_first',
        separate_edge_classes=True)
    np.testing.assert_equal(mask_transform.shape, (5, 4, 10, 30, 30))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

def test_outer_distance_transform():
    test_dtype = np.float16
    # test 2D masks
    distance_bins = None
    erosion_width = 1
    mask = np.random.randint(3, size=(5, 30, 30, 1))
    mask_transform = _transform_masks(
        mask,
        transform='outer-distance',
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        data_format='channels_last',
        mask_dtype = test_dtype
    )
    np.testing.assert_equal(mask_transform.shape, (5, 30, 30, 1))
    np.testing.assert_equal(mask_transform.dtype, test_dtype)

    distance_bins = 4
    erosion_width = 1
    mask = np.random.randint(3, size=(5, 30, 30, 1))
    mask_transform = _transform_masks(
        mask,
        transform='outer-distance',
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        data_format='channels_last',
        mask_dtype = test_dtype
    )
    np.testing.assert_equal(mask_transform.shape, (5, 30, 30, distance_bins))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    distance_bins = 6
    mask = np.random.randint(3, size=(5, 1, 30, 30))
    mask_transform = _transform_masks(
        mask,
        transform='outer-distance',
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        data_format='channels_first',
        mask_dtype = test_dtype
    )
    np.testing.assert_equal(mask_transform.shape, (5, distance_bins, 30, 30))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    # test 3D masks
    test_dtype = np.float32
    distance_bins = None
    mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
    mask_transform = _transform_masks(
        mask,
        transform='outer-distance',
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        data_format='channels_last',
        mask_dtype = test_dtype
    )
    np.testing.assert_equal(mask_transform.shape, (5, 10, 30, 30, 1))
    np.testing.assert_equal(mask_transform.dtype, test_dtype)

    distance_bins = 5
    mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
    mask_transform = _transform_masks(
        mask,
        transform='outer-distance',
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        data_format='channels_last',
        mask_dtype = test_dtype
    )
    np.testing.assert_equal(mask_transform.shape, (5, 10, 30, 30, distance_bins))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    distance_bins = 4
    mask = np.random.randint(3, size=(5, 1, 10, 30, 30))
    mask_transform = _transform_masks(
        mask,
        transform='outer-distance',
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        data_format='channels_first',
        mask_dtype = test_dtype
    )
    np.testing.assert_equal(mask_transform.shape, (5, distance_bins, 10, 30, 30))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

def test_inner_distance_transform():
    test_dtype = np.float16

    # test 2D masks
    distance_bins = None
    erosion_width = 1
    mask = np.random.randint(3, size=(5, 30, 30, 1))
    mask_transform = _transform_masks(
        mask,
        transform='inner-distance',
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        data_format='channels_last',
        mask_dtype = test_dtype
    )
    np.testing.assert_equal(mask_transform.shape, (5, 30, 30, 1))
    np.testing.assert_equal(mask_transform.dtype, test_dtype)

    distance_bins = 4
    erosion_width = 1
    mask = np.random.randint(3, size=(5, 30, 30, 1))
    mask_transform = _transform_masks(
        mask,
        transform='inner-distance',
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        data_format='channels_last',
        mask_dtype = test_dtype
    )
    np.testing.assert_equal(mask_transform.shape, (5, 30, 30, distance_bins))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    distance_bins = 6
    mask = np.random.randint(3, size=(5, 1, 30, 30))
    mask_transform = _transform_masks(
        mask,
        transform='inner-distance',
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        data_format='channels_first',
        mask_dtype = test_dtype
    )
    np.testing.assert_equal(mask_transform.shape, (5, distance_bins, 30, 30))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    # test 3D masks
    test_dtype = np.float32
    distance_bins = None
    mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
    mask_transform = _transform_masks(
        mask,
        transform='inner-distance',
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        data_format='channels_last',
        mask_dtype = test_dtype
    )
    np.testing.assert_equal(mask_transform.shape, (5, 10, 30, 30, 1))
    np.testing.assert_equal(mask_transform.dtype, test_dtype)

    distance_bins = 5
    mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
    mask_transform = _transform_masks(
        mask,
        transform='inner-distance',
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        data_format='channels_last',
        mask_dtype = test_dtype
    )
    np.testing.assert_equal(mask_transform.shape, (5, 10, 30, 30, distance_bins))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

    distance_bins = 4
    mask = np.random.randint(3, size=(5, 1, 10, 30, 30))
    mask_transform = _transform_masks(
        mask,
        transform='inner-distance',
        distance_bins=distance_bins,
        erosion_width=erosion_width,
        data_format='channels_first',
        mask_dtype = test_dtype
    )
    np.testing.assert_equal(mask_transform.shape, (5, distance_bins, 10, 30, 30))
    assert(np.issubdtype(mask_transform.dtype, np.integer))

def test_disc_transform():
    test_dtype = np.float32
    classes = np.random.randint(5, size=1)[0]
    classes = max(1, classes)
    # test 2D masks
    mask = np.random.randint(classes, size=(5, 30, 30, 1))
    mask_transform = _transform_masks(
        mask,
        transform='disc',
        data_format='channels_last')
    np.testing.assert_equal(mask_transform.shape, (5, 30, 30, classes))
    np.testing.assert_equal(mask_transform.dtype, test_dtype)

    mask = np.random.randint(classes, size=(5, 1, 30, 30))
    mask_transform = _transform_masks(
        mask,
        transform='disc',
        data_format='channels_first')
    np.testing.assert_equal(mask_transform.shape, (5, classes, 30, 30))
    np.testing.assert_equal(mask_transform.dtype, test_dtype)

    # test 3D masks
    mask = np.random.randint(classes, size=(5, 10, 30, 30, 1))
    mask_transform = _transform_masks(
        mask,
        transform='disc',
        data_format='channels_last')
    np.testing.assert_equal(mask_transform.shape, (5, 10, 30, 30, classes))
    np.testing.assert_equal(mask_transform.dtype, test_dtype)

    mask = np.random.randint(classes, size=(5, 1, 10, 30, 30))
    mask_transform = _transform_masks(
        mask,
        transform='disc',
        data_format='channels_first')
    np.testing.assert_equal(mask_transform.shape, (5, classes, 10, 30, 30))
    np.testing.assert_equal(mask_transform.dtype, test_dtype)

def test_bad_mask():
    # test bad transform
    with pytest.raises(ValueError):
        mask = np.random.randint(3, size=(5, 30, 30, 1))
        _transform_masks(mask, transform='unknown')

    # test bad channel axis 2D
    with pytest.raises(ValueError):
        mask = np.random.randint(3, size=(5, 30, 30, 2))
        _transform_masks(mask, transform=None)

    # test bad channel axis 3D
    with pytest.raises(ValueError):
        mask = np.random.randint(3, size=(5, 10, 30, 30, 2))
        _transform_masks(mask, transform=None)

    # test ndim < 4
    with pytest.raises(ValueError):
        mask = np.random.randint(3, size=(5, 30, 1))
        _transform_masks(mask, transform=None)

    # test ndim > 5
    with pytest.raises(ValueError):
        mask = np.random.randint(3, size=(5, 10, 30, 30, 10, 1))
        _transform_masks(mask, transform=None)