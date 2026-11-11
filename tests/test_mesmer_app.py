"""End-to-end tests of ``torch_mesmer.mesmer.Mesmer.predict``.

Replaces the old ``torch_mesmer/mesmer_test.py``, which imported a module that
no longer exists (``.panoptic``) and passed channels-last images.

``Mesmer.__init__`` always loads released weights (and downloads them if
missing), so these tests build the app around a small stand-in network instead.
The stand-in has the real model's input/output contract, so everything else in
``predict`` (resize, preprocess, tile, untile, watershed, resize back) runs for
real. One ``slow`` test builds the actual PanopticNet.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from torch_mesmer import mesmer as mesmer_module
from torch_mesmer.mesmer import Mesmer
from torch_mesmer.model import PanopticNet


class StubPanopticNet(nn.Module):
    """(B, 2, H, W) -> (B, 8, H, W) in the port's head layout: ReLU on the two
    inner-distance heads (channels 0 and 4) and softmax on the two 3-class
    pixelwise heads (1-3 and 5-7)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(2, 8, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv(x)
        return torch.cat([
            torch.relu(y[:, 0:1]), torch.softmax(y[:, 1:4], dim=1),
            torch.relu(y[:, 4:5]), torch.softmax(y[:, 5:8], dim=1),
        ], dim=1)


def make_app(model):
    """A ``Mesmer`` with the attributes ``__init__`` sets, minus weight loading."""
    app = Mesmer.__new__(Mesmer)
    app.device = 'cpu'
    app.model = model.eval()
    app.model_image_shape = (256, 256)
    app.required_rank = 4
    app.required_channels = 2
    app.model_mpp = 0.5
    return app


@pytest.fixture(scope='module')
def app():
    return make_app(StubPanopticNet())


@pytest.fixture
def image():
    """Random 2-channel image, channels-first (B, C, H, W): nuclear, then membrane."""
    return np.random.default_rng(0).random((1, 2, 200, 200)).astype(np.float32)


@pytest.mark.parametrize('mpp', [0.375, 0.5, 0.75])  # below, at and above the model's 0.5
def test_mask_shape_resizing(app, image, mpp):
    mask = app.predict(image, image_mpp=mpp)
    assert mask.shape == (1, 1, 200, 200)


@pytest.mark.parametrize('mpp', [0.375, 0.5, 0.75])
@pytest.mark.parametrize('compartment, n_out', [('nuclear', 1), ('whole-cell', 1), ('both', 2)])
def test_compartments(app, image, mpp, compartment, n_out):
    mask = app.predict(image, image_mpp=mpp, compartment=compartment)
    assert mask.shape == (1, n_out, 200, 200)
    assert np.issubdtype(mask.dtype, np.integer)


def test_large_image_is_tiled(app):
    image = np.random.default_rng(1).random((2, 2, 600, 520)).astype(np.float32)
    mask, transforms = app.predict(image, image_mpp=0.5, compartment='both', return_transforms=True)
    assert mask.shape == (2, 2, 600, 520)
    assert transforms.shape == (2, 8, 600, 520)


def test_channels_last_input_is_rejected(app):
    """A (batch, H, W, channels) array raises an error that says to ``moveaxis``."""
    with pytest.raises(ValueError, match='moveaxis'):
        app.predict(np.zeros((1, 256, 256, 2), np.float32), image_mpp=0.5)


@pytest.mark.parametrize('shape', [(2, 256, 256), (1, 3, 256, 256), (1, 1, 2, 256, 256)])
def test_wrong_rank_or_channels_is_rejected(app, shape):
    """Only (batch, 2, H, W) input is accepted."""
    with pytest.raises(ValueError, match='Input data must have'):
        app.predict(np.zeros(shape, np.float32), image_mpp=0.5)


@pytest.mark.parametrize('shape', [(1, 2, 200, 600), (1, 2, 600, 200)])
def test_one_side_small_one_side_large(app, shape):
    """An image shorter than 256 px on one side and longer on the other still segments."""
    image = np.random.default_rng(2).random(shape).astype(np.float32)
    mask, transforms = app.predict(image, image_mpp=0.5, compartment='both', return_transforms=True)
    assert mask.shape == (1, 2) + shape[2:]
    assert transforms.shape == (1, 8) + shape[2:]


def test_init_skips_imagenet_download(monkeypatch):
    """Mesmer() builds the network without ImageNet weights, then loads the checkpoint."""
    built = {}

    def fake_panopticnet(**kwargs):
        built.update(kwargs)
        return StubPanopticNet()

    monkeypatch.setattr(mesmer_module, 'PanopticNet', fake_panopticnet)
    monkeypatch.setattr(mesmer_module.torch, 'load', lambda *a, **k: StubPanopticNet().state_dict())
    app = Mesmer(model_path='unused.pth')
    assert built['use_imagenet'] is False
    assert not app.model.training


def _fresh_interpreter(code):
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_import_does_not_change_torch_threads():
    """Importing ``torch_mesmer.mesmer`` leaves the caller's torch thread count alone."""
    n = _fresh_interpreter(
        'import torch; torch.set_num_threads(3); import torch_mesmer.mesmer; print(torch.get_num_threads())')
    assert n == '3'


def test_utils_import_does_not_load_matplotlib():
    """Importing ``torch_mesmer.utils`` does not pull in matplotlib (slow, needs a display)."""
    loaded = _fresh_interpreter("import sys, torch_mesmer.utils; print('matplotlib' in sys.modules)")
    assert loaded == 'False'


@pytest.mark.slow
def test_panopticnet_forward_contract():
    """The real network: output layout and head activations."""
    torch.manual_seed(0)
    model = PanopticNet(use_imagenet=False).eval()
    with torch.inference_mode():
        out = model(torch.rand(1, 2, 256, 256))
    assert out.shape == (1, 8, 256, 256)
    for start in (1, 5):  # pixelwise heads are softmax over 3 classes
        torch.testing.assert_close(out[:, start:start + 3].sum(dim=1), torch.ones(1, 256, 256))
    for ch in (0, 4):     # inner-distance heads are non-negative
        assert (out[:, ch] >= 0).all()
