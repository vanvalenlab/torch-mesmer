import numpy as np
import pytest

import torch

from torch_mesmer.mesmer import Mesmer

from _utils import requires_gpu, requires_model


@pytest.fixture(scope="module")
def default_app():
    app = Mesmer()
    return app


@pytest.fixture()
def random_img():
    """Generate a random 2-channel image."""
    return np.random.random((2, 100, 100))


@requires_model
def test_api_image_mpp_required(default_app, random_img):
    """image_mpp is a required argument for .predict"""
    with pytest.raises(TypeError, match="missing.*required positional argument"):
        default_app.predict(random_img)


@requires_gpu
@requires_model
@pytest.mark.parametrize("mpp", (0.375, 0.5, 0.75))  # lt default, default, gt default
def test_mask_shape_resizing(default_app, random_img, mpp):
    """Check that mask resizing is done properly."""
    img = random_img
    mask = default_app.predict(img[np.newaxis, ...], image_mpp=mpp)
    assert img.shape[1:] == mask.squeeze().shape


@requires_gpu
@requires_model
@pytest.mark.parametrize("mpp", (0.375, 0.5, 0.75))
@pytest.mark.parametrize("compartment", ("nuclear", "whole-cell", "both"))
def test_compartments(default_app, random_img, mpp, compartment):
    img = random_img
    mask = default_app.predict(
        img[np.newaxis, ...], image_mpp=mpp, compartment=compartment
    )
    wh = img.shape[1:]
    expected_shape = (2, *wh) if compartment == "both" else wh
    assert mask.squeeze().shape == expected_shape
