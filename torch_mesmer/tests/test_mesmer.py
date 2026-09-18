import numpy as np
import pytest

import torch

from .panoptic import PanopticNet
from .mesmer import Mesmer


@pytest.fixture(scope="module")
def default_app():
    with torch.no_grad():
        model = PanopticNet(
            'resnet50',
            input_shape=(256, 256, 2),
            norm_method=None,
            num_semantic_heads=4,
            num_semantic_classes=[1, 3, 1, 3],
            location=True,
            include_top=True,
            use_imagenet=False,
        )
        app = Mesmer(model, device="cpu")
    return app


@pytest.fixture()
def random_img():
    """Generate a random 2-channel image."""
    return np.random.random((200, 200, 2))


@pytest.mark.parametrize("mpp", (0.375, 0.5, 0.75))  # lt default, default, gt default
def test_mask_shape_resizing(default_app, random_img, mpp):
    """Check that mask resizing is done properly."""
    img = random_img
    mask = default_app.predict(img[np.newaxis, ...], image_mpp=mpp)
    assert img.shape[:-1] == mask.squeeze().shape


@pytest.mark.parametrize("mpp", (0.375, 0.5, 0.75))
@pytest.mark.parametrize("compartment", ("nuclear", "whole-cell", "both"))
def test_compartments(default_app, random_img, mpp, compartment):
    img = random_img
    mask = default_app.predict(
        img[np.newaxis, ...], image_mpp=mpp, compartment=compartment
    )
    wh = img.shape[:-1]
    expected_shape = (*wh, 2) if compartment == "both" else wh
    assert mask.squeeze().shape == expected_shape
