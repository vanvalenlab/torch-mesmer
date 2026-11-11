"""Architecture tests for ``PanopticNet`` and its modules.

The FPN wiring tests use a tiny pyramid and write the expected computation out
by hand, following deepcell-tf ``model_zoo/fpn.py``. The full-network tests
build the real model on CPU (a few seconds each).
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch_mesmer.model import ARCH_VERSION, PanopticNet, checkpoint_arch_version
from torch_mesmer.modules import BackboneNetwork, FeaturePyramidNetwork

RELEASED_WEIGHTS = Path.home() / '.deepcell' / 'models' / 'torch-mesmer_2026-06-30.pth'

# Parameter counts, recorded so that an unintended architecture change shows up.
# Version 1 (released 2026-06-30 weights) has 87,832,151. Version 2 adds 4,128,768:
# P6 reads C5 (2048 channels) instead of P5 (256), and its 3x3 conv has
# (2048 - 256) * 256 * 9 more weights.
N_PARAMS_V2 = 91_960_919
N_PARAMS_V1 = 87_832_151


def up(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


@pytest.fixture
def tiny_pyramid():
    torch.manual_seed(0)
    channels = {'C3': 8, 'C4': 16, 'C5': 32}
    feats = {'C3': torch.randn(2, 8, 16, 16), 'C4': torch.randn(2, 16, 8, 8), 'C5': torch.randn(2, 32, 4, 4)}
    return channels, feats


def build_fpn(channels, legacy):
    torch.manual_seed(1)
    return FeaturePyramidNetwork(levels=['P3', 'P4', 'P5', 'P6', 'P7'], backbone_levels=['C3', 'C4', 'C5'],
                                 feature_size=4, backbone_channels=channels, legacy=legacy)


def test_fpn_top_down_passes_merged_lateral(tiny_pyramid):
    channels, c = tiny_pyramid
    fpn = build_fpn(channels, legacy=False)
    lat = {k: fpn.levels[k].lateral_conv for k in ('P3', 'P4', 'P5')}
    smooth = {k: fpn.levels[k].smooth_conv for k in ('P3', 'P4', 'P5')}
    with torch.no_grad():
        out = fpn(c)
        m5 = lat['P5'](c['C5'])
        m4 = lat['P4'](c['C4']) + up(m5)
        m3 = lat['P3'](c['C3']) + up(m4)
        p6 = fpn.crown['P6'](c['C5'])                  # P6 comes from C5
        expected = {'P3': smooth['P3'](m3), 'P4': smooth['P4'](m4), 'P5': smooth['P5'](m5),
                    'P6': p6, 'P7': fpn.crown['P7'](torch.relu(p6))}  # ReLU before P7
    assert out.keys() == expected.keys()
    for k in expected:
        torch.testing.assert_close(out[k], expected[k], msg=k)


def test_fpn_legacy_wiring(tiny_pyramid):
    """legacy=True reproduces the original network the released weights were trained with."""
    channels, c = tiny_pyramid
    fpn = build_fpn(channels, legacy=True)
    lat = {k: fpn.levels[k].lateral_conv for k in ('P3', 'P4', 'P5')}
    smooth = {k: fpn.levels[k].smooth_conv for k in ('P3', 'P4', 'P5')}
    with torch.no_grad():
        out = fpn(c)
        p5 = smooth['P5'](lat['P5'](c['C5']))
        p4 = smooth['P4'](lat['P4'](c['C4']) + up(p5))
        p3 = smooth['P3'](lat['P3'](c['C3']) + up(p4))
        p6 = fpn.crown['P6'](p5)
        p7 = fpn.crown['P7'](p6)
    for k, v in dict(P3=p3, P4=p4, P5=p5, P6=p6, P7=p7).items():
        torch.testing.assert_close(out[k], v, msg=k)


def test_crown_levels_sorted_numerically():
    channels = {'C3': 8, 'C4': 16, 'C5': 32}
    levels = [f'P{i}' for i in range(3, 11)]
    fpn = FeaturePyramidNetwork(levels=levels, backbone_channels=channels, feature_size=4)
    assert fpn.crown_levels == ['P6', 'P7', 'P8', 'P9', 'P10']


def test_crown_levels_must_be_contiguous():
    with pytest.raises(ValueError, match='contiguous'):
        FeaturePyramidNetwork(levels=['P3', 'P4', 'P5', 'P7'], backbone_channels={'C3': 8, 'C4': 16, 'C5': 32})


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    return PanopticNet(use_imagenet=False).eval()


@pytest.mark.slow
def test_parameter_count(model):
    assert sum(p.numel() for p in model.parameters()) == N_PARAMS_V2
    legacy = PanopticNet(use_imagenet=False, legacy=True)
    assert sum(p.numel() for p in legacy.parameters()) == N_PARAMS_V1


@pytest.mark.slow
@pytest.mark.parametrize('in_channels', [1, 3])
def test_in_channels_honoured(in_channels):
    """The 1x1 channel conv takes image + 2 location channels and is always applied."""
    net = PanopticNet(use_imagenet=False, in_channels=in_channels, crop_size=128).eval()
    assert net.channel_conv.in_channels == in_channels + 2
    calls = []
    net.channel_conv.register_forward_hook(lambda *a: calls.append(1))
    with torch.inference_mode():
        out = net(torch.rand(1, in_channels, 128, 128))
    assert calls == [1]
    assert out.shape == (1, 8, 128, 128)


@pytest.mark.slow
def test_checkpoint_version_guard(tmp_path):
    net = PanopticNet(use_imagenet=False, crop_size=128)
    sd = net.state_dict()
    assert checkpoint_arch_version(sd) == ARCH_VERSION
    torch.save(sd, tmp_path / 'w.pth')
    PanopticNet(use_imagenet=False, crop_size=128).load_state_dict(torch.load(tmp_path / 'w.pth'))
    with pytest.raises(ValueError, match='legacy=False'):
        PanopticNet(use_imagenet=False, crop_size=128, legacy=True).load_state_dict(sd)
    sd['arch_version'] = torch.tensor(ARCH_VERSION + 1)  # a version this code does not build
    with pytest.raises(ValueError, match='does not build'):
        PanopticNet(use_imagenet=False, crop_size=128).load_state_dict(sd)


@pytest.mark.slow
@pytest.mark.skipif(not RELEASED_WEIGHTS.exists(), reason='released weights not found')
def test_released_weights_need_legacy():
    sd = torch.load(RELEASED_WEIGHTS, map_location='cpu')
    assert checkpoint_arch_version(sd) == 1
    PanopticNet(use_imagenet=False, legacy=True).load_state_dict(sd)
    with pytest.raises(ValueError, match='legacy=True'):
        PanopticNet(use_imagenet=False).load_state_dict(sd)


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
