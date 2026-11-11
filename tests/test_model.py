"""Architecture tests for ``PanopticNet`` and its modules.

The full-network tests build the real model on CPU (a few seconds each).
"""

import pytest
import torch
from torch_mesmer.model import PanopticNet
from torch_mesmer.modules import BackboneNetwork

# --- EfficientNetV2-L backbone ---

@pytest.fixture(scope='module')
def effnet():
    torch.manual_seed(0)
    return BackboneNetwork('efficientnetv2bl', use_imagenet=False).eval()


@pytest.mark.slow
def test_efficientnet_taps(effnet):
    """Taps match deepcell's Keras layers: block1b_add, block2c_add, block4a_expand_activation,
    block6a_expand_activation, top_activation (backbone_utils.py:418-432)."""
    assert effnet.out_channels == {'C1': 32, 'C2': 64, 'C3': 384, 'C4': 1344, 'C5': 1280}
    with torch.inference_mode():
        feats = effnet(torch.rand(1, 3, 128, 128))
    assert {k: v.shape[-1] for k, v in feats.items()} == {'C1': 64, 'C2': 32, 'C3': 16, 'C4': 8, 'C5': 4}


@pytest.mark.slow
def test_efficientnet_split_is_exact(effnet):
    """Cutting MBConv blocks at their expand conv must not change the network: C5 equals
    torchvision's own ``features`` forward on the same modules."""
    from torchvision.models import efficientnet_v2_l
    torch.manual_seed(0)
    reference = efficientnet_v2_l().eval()
    torch.manual_seed(0)
    ours = BackboneNetwork('efficientnetv2bl', use_imagenet=False).eval()
    ours_sd = {k: v for k, v in ours.state_dict().items()}
    assert sum(v.numel() for v in ours_sd.values()) == sum(
        v.numel() for k, v in reference.state_dict().items() if k.startswith('features.'))
    x = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        torch.testing.assert_close(ours(x)['C5'], reference.features(x))


@pytest.mark.slow
def test_efficientnet_panopticnet_forward():
    net = PanopticNet(backbone='efficientnetv2bl', use_imagenet=False, crop_size=128).eval()
    with torch.inference_mode():
        out = net(torch.rand(1, 2, 128, 128))
    assert out.shape == (1, 8, 128, 128)
