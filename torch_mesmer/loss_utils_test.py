import pytest

import torch

from .loss_utils import weighted_categorical_crossentropy

ALL_LOSSES = [
    # losses.categorical_crossentropy,
    weighted_categorical_crossentropy,
    # losses.sample_categorical_crossentropy,
    # losses.weighted_focal_loss,
    # losses.smooth_l1,
    # losses.focal,
    # losses.dice_loss,
    # losses.discriminative_instance_loss
]

def test_objective_shapes_3d():
    y_a = torch.rand((5, 6, 7))
    y_b = torch.rand((5, 6, 7))
    for obj in ALL_LOSSES:
        objective_output = obj(y_a, y_b, axis=-1)
        assert(list(objective_output.shape) == [5, 6])
        objective_output = obj(y_a, y_b, axis=1)
        assert(list(objective_output.shape) == [5, 7])

def test_objective_shapes_2d():
    y_a = torch.rand(6, 7)
    y_b = torch.rand(6, 7)
    for obj in ALL_LOSSES:
        objective_output = obj(y_a, y_b, axis=-1)
        assert(list(objective_output.shape) == [6])