import pytest 

import torch

from .panoptic import PanopticNet

@pytest.mark.parametrize("pooling, location, frames_per_batch, data_format, upsample_type, pyramid_levels", 
                         [(None, False, 1, "channels_last", "upsamplelike", ["P3"]), # Will fail as "upsamplelike" is not incorporated
                          (None, True, 1, "channels_first", "upsampling2d", ['P3', 'P4', 'P5', 'P6', 'P7'])])
def test_panopticnet(pooling, location, frames_per_batch,
                         data_format, upsample_type, pyramid_levels):
    norm_method = None

    # not all backbones work with channels_first
    backbone = 'resnet50'

    # TODO: PanopticNet fails with channels_first and frames_per_batch > 1
    if frames_per_batch > 1 and data_format == 'channels_first':
        return

    
    # K.set_image_data_format(data_format)
    # if data_format == 'channels_first':
    #     axis = 1
    #     input_shape = (1, 32, 32)
    # else:
    #     axis = -1
    #     input_shape = (32, 32, 1)

    input_shape = (256, 256, 2)
    axis = 1
    
    num_semantic_classes = [1, 3]

    # temporal_mode=None,
    # lite=False,
    # interpolation='bilinear',

    model = PanopticNet(
        backbone=backbone,
        input_shape=input_shape,
        frames_per_batch=frames_per_batch,
        pyramid_levels=pyramid_levels,
        norm_method=norm_method,
        location=location,
        pooling=pooling,
        upsample_type=upsample_type,
        num_semantic_classes=num_semantic_classes,
        use_imagenet=False,
    )

    input = torch.rand(1, 2, 256, 256)
    output = model(input)

    assert isinstance(output, list)
    assert(len(output) == len(num_semantic_classes))
    for i, s in enumerate(num_semantic_classes):
        assert(output[i].shape[axis] == s)

def test_panopticnet_semantic_class_types():
    shared_kwargs = {
        'backbone': 'resnet50',
        'input_shape': (32, 32, 1),
        'use_imagenet': False,
    }

    nsc1 = [2, 3]
    model1 = PanopticNet(num_semantic_classes=nsc1, **shared_kwargs)

    nsc2 = {'0': 2, '1': 3}
    model2 = PanopticNet(num_semantic_classes=nsc2, **shared_kwargs)

    inputs = torch.rand(3, 2, 256, 256)
    outputs1 = model1(inputs)
    outputs2 = model2(inputs)

    for o1, o2 in zip(outputs1, outputs2):
        assert(list(o1.shape) == list(o2.shape))
        assert(o1.dtype == o2.dtype)


# Will fail as input hardcoded to be (2, 256, 256)
# Tested with commented checks in panoptic.py, so test is fine
# Just needed to reimplement in panoptic.py
def test_panopticnet_bad_input():
    norm_method = None

    # not all backbones work with channels_first
    backbone = 'resnet50'

    num_semantic_classes = [1, 3]

    # non-square input
    input_shape = (256, 512, 1)
    with pytest.raises(ValueError):
        PanopticNet(
            backbone=backbone,
            input_shape=input_shape,
            backbone_levels=['C3', 'C4', 'C5'],
            norm_method=norm_method,
            location=True,
            pooling=None,
            num_semantic_classes=num_semantic_classes,
            use_imagenet=False,
        )

    # non power of 2 input
    input_shape = (257, 257, 1)
    with pytest.raises(ValueError):
        PanopticNet(
            backbone=backbone,
            input_shape=input_shape,
            backbone_levels=['C3', 'C4', 'C5'],
            norm_method=norm_method,
            location=True,
            pooling=None,
            num_semantic_classes=num_semantic_classes,
            use_imagenet=False,
        )
