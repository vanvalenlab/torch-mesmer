"""Tests for post-processing functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pytest

from torch_mesmer.deep_watershed import deep_watershed


def test_deep_watershed():
    shape = (5, 21, 21, 1)
    maxima = np.random.random(shape) * 100
    interior = np.random.random(shape) * 100
    other = np.random.randint(0, 1, size=shape)
    inputs = [maxima, interior]

    # basic tests for both h_maxima and peak_local_max
    for algo in ('h_maxima', 'peak_local_max'):
        label_img = deep_watershed(inputs, maxima_algorithm=algo)
        np.testing.assert_equal(label_img.shape, shape[:-1] + (1,))

        # flip the order and give correct indices, same answer
        label_img_2 = deep_watershed([other, maxima, interior],
                                                    maxima_index=1,
                                                    interior_index=2,
                                                    maxima_algorithm=algo)
        np.testing.assert_array_equal(label_img, label_img_2)

        # all the bells and whistles
        label_img_3 = deep_watershed(inputs, maxima_algorithm=algo,
                                                    small_objects_threshold=1,
                                                    label_erosion=1,
                                                    pixel_expansion=1,
                                                    fill_holes_threshold=1)

        np.testing.assert_equal(label_img_3.shape, shape[:-1] + (1,))

    # test bad inputs, pairs of maxima and interior shapes
    bad_shapes = [
        ((1, 32, 32, 1), (1, 32, 16, 1)),  # unequal dimensions
        ((1, 32, 32, 1), (1, 16, 32, 1)),  # unequal dimensions
        ((32, 32, 1), (32, 32, 1)),  # no batch dimension
        ((1, 32, 32), (1, 32, 32)),  # no channel dimension
        ((1, 5, 10, 32, 32, 1), (1, 5, 10, 32, 32, 1)),  # too many dims
    ]
    for bad_maxima_shape, bad_interior_shape in bad_shapes:
        bad_inputs = [np.random.random(bad_maxima_shape),
                      np.random.random(bad_interior_shape)]
        with pytest.raises(ValueError):
            deep_watershed(bad_inputs)

    # test bad values of maxima_algorithm.
    with pytest.raises(ValueError):
        deep_watershed(inputs, maxima_algorithm='invalid')

    # pass weird data types
    bad_inputs = [
        {'interior-distance': maxima, 'outer-distance': interior},
        None,
    ]
    for bad_input in bad_inputs:
        with pytest.raises(ValueError):
            _ = deep_watershed(bad_input)

    # test deprecated values still work
    # each pair is the deprecated name, then the new name.
    old_new_pairs = [
        ('min_distance', 'radius', np.random.randint(10)),
        ('distance_threshold', 'interior_threshold', np.random.randint(1, 100) / 100),
        ('detection_threshold', 'maxima_threshold', np.random.randint(1, 100) / 100),
    ]
    for deprecated_arg, new_arg, value in old_new_pairs:
        dep_kwargs = {deprecated_arg: value}
        new_kwargs = {new_arg: value}

        with pytest.deprecated_call():
            dep_img = deep_watershed(inputs, **dep_kwargs)
        new_img = deep_watershed(inputs, **new_kwargs)
        np.testing.assert_array_equal(dep_img, new_img)
