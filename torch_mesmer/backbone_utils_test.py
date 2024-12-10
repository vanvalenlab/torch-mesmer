import pytest
import torch

from .backbone_utils import get_backbone

def test_get_backbone():
    backbone = "resnet50"
    inputs = torch.nn.Identity()
    model, output_dict = get_backbone(backbone, inputs, return_dict=True)
    assert isinstance(output_dict, dict)
    assert all(k.startswith('C') for k in output_dict)
    assert isinstance(model, torch.nn.Module)

def test_invalid_backbone():
    backbone = "resnet50"
    inputs = torch.nn.Identity()
    with pytest.raises(ValueError):
        get_backbone('bad', inputs, return_dict=True)

def test_invalid_backbone_input():
    backbone = "resnet50"
    bad_inputs = torch.rand([1, 2, 3, 4])
    with pytest.raises(TypeError):
        get_backbone(backbone, bad_inputs, return_dict=True)