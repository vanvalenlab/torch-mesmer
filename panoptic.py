import math
import numpy as np
import re

import torch
import torch.nn as nn
from torch.nn import Conv2d, Conv3d, LazyConv2d


from fpn import __create_pyramid_features
from fpn import __create_semantic_head
from layers import Location2D
from backbone_utils import get_backbone


class combine_models(nn.Module):
    def __init__(self, model_li, device):
        super().__init__()
        self.model_li = nn.ModuleList(model_li)
        self.input_shape = (None, 256, 256, 2)
        self.device = device
        # self.input_shape = (None, 2, 256, 256)

    def predict(self, batch_input, batch_size):
        assert(len(batch_input)<=batch_size)

        temp_input = np.transpose(batch_input, (0, 3, 1, 2))
        outs = self.forward(temp_input)
        outs_torch = [torch.permute(i, (0, 2, 3, 1)) for i in outs]

        return outs_torch
            
    
    def forward(self, input):
        temp_input = input
        if not torch.is_tensor(temp_input):
            temp_input = torch.tensor(temp_input)
            
        if torch.cuda.is_available():
            temp_input = temp_input.to(self.device)

        model_out_li = [self.model_li[i](temp_input) for i in range(len(self.model_li))]
        # model_out_li = [nn.Parameter(i) for i in model_out_li]
        
        return model_out_li

class pls_concat_properly(nn.Module):
    def __init__(self, path1, path2):
        super(pls_concat_properly, self).__init__()
        # might be unnecessary
        self.path_li = nn.ModuleList([path1, path2])
    
    def forward(self, input):
        
        return torch.cat([self.path_li[0](input), self.path_li[1](input)], dim=1)

def PanopticNet(backbone,
                input_shape,
                inputs=None,
                backbone_levels=['C3', 'C4', 'C5'],
                pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
                create_pyramid_features=__create_pyramid_features,
                create_semantic_head=__create_semantic_head,
                frames_per_batch=1,
                temporal_mode=None,
                num_semantic_classes=[3],
                required_channels=3,
                norm_method=None,
                pooling=None,
                location=True,
                use_imagenet=True,
                lite=False,
                upsample_type='upsampling2d',
                interpolation='bilinear',
                name='panopticnet',
                z_axis_convolutions=False,
                device=torch.device("cpu"),
                **kwargs):
    """Constructs a Mask-RCNN model using a backbone from
    ``keras-applications`` with optional semantic segmentation transforms.

    Args:
        backbone (str): Name of backbone to use.
        input_shape (tuple): The shape of the input data.
        backbone_levels (list): The backbone levels to be used.
            to create the feature pyramid.
        pyramid_levels (list): Pyramid levels to use.
        create_pyramid_features (function): Function to get the pyramid
            features from the backbone.
        create_semantic_head (function): Function to build a semantic head
            submodel.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.
        temporal_mode: Mode of temporal convolution. Choose from
            ``{'conv','lstm', None}``.
        num_semantic_classes (list or dict): Number of semantic classes
            for each semantic head. If a ``dict``, keys will be used as
            head names and values will be the number of classes.
        norm_method (str): Normalization method to use with the
            :mod:`deepcell.layers.normalization.ImageNormalization2D` layer.
        location (bool): Whether to include a
            :mod:`deepcell.layers.location.Location2D` layer.
        use_imagenet (bool): Whether to load imagenet-based pretrained weights.
        lite (bool): Whether to use a depthwise conv in the feature pyramid
            rather than regular conv.
        upsample_type (str): Choice of upsampling layer to use from
            ``['upsamplelike', 'upsampling2d', 'upsampling3d']``.
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ``['bilinear', 'nearest']``.
        pooling (str): optional pooling mode for feature extraction
            when ``include_top`` is ``False``.

            - None means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
            - 'avg' means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
            - 'max' means that global max pooling will
              be applied.

        z_axis_convolutions (bool): Whether or not to do convolutions on
            3D data across the z axis.
        required_channels (int): The required number of channels of the
            backbone.  3 is the default for all current backbones.
        kwargs (dict): Other standard inputs for ``retinanet_mask``.

    Raises:
        ValueError: ``temporal_mode`` not 'conv', 'lstm'  or ``None``

    Returns:
        tensorflow.keras.Model: Panoptic model with a backbone.
    """
    
    # channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # EDIT: using channels last as that's the default for cropping generator
    # might have to change to channels first as pytorch prefers channels_first
    input_shape2 = input_shape
    input_shape = (2, 256, 256)
    channel_axis = 1

    # conv = Conv3D if frames_per_batch > 1 else Conv2D
    # EDIT: using pytorch versions
    conv = Conv3d if frames_per_batch > 1 else LazyConv2d
    conv_kernel = (1, 1, 1) if frames_per_batch > 1 else (1, 1)

    # Check input to __merge_temporal_features
    acceptable_modes = {'conv', 'lstm', None}
    if temporal_mode is not None:
        temporal_mode = str(temporal_mode).lower()
        if temporal_mode not in acceptable_modes:
            raise ValueError(f'temporal_mode {temporal_mode} not supported. Please choose '
                             f'from {acceptable_modes}.')

    # TODO only works for 2D: do we check for 3D as well?
    # What are the requirements for 3D data?
    img_shape = input_shape[1:] if channel_axis == 1 else input_shape[:-1]
    if img_shape[0] != img_shape[1]:
        raise ValueError(f'Input data must be square, got dimensions {img_shape}')

    if not math.log(img_shape[0], 2).is_integer():
        raise ValueError('Input data dimensions must be a power of 2, '
                         f'got {img_shape[0]}')

    # Check input to interpolation
    acceptable_interpolation = {'bilinear', 'nearest'}
    if interpolation not in acceptable_interpolation:
        raise ValueError(f'Interpolation mode "{interpolation}" not supported. '
                         f'Choose from {list(acceptable_interpolation)}.')

    if inputs is None:
        if frames_per_batch > 1:
            if channel_axis == 1:
                input_shape_with_time = tuple(
                    [input_shape[0], frames_per_batch] + list(input_shape)[1:])
            else:
                input_shape_with_time = tuple(
                    [frames_per_batch] + list(input_shape))
            inputs = Input(shape=input_shape_with_time, name='input_0')
        else:
            # inputs2 = Input(shape=input_shape2, name='input_0')
            # EDIT: pytorch
            temp_input_shape = [1] + list(input_shape)
            inputs = torch.randn(size=temp_input_shape)
            

    # Normalize input images
    if norm_method is None:
        # norm = inputs
        norm = nn.Identity()
    else:
        if frames_per_batch > 1:
            norm = TimeDistributed(ImageNormalization2D(
                norm_method=norm_method, name='norm'), name='td_norm')(inputs)
        else:
            norm = ImageNormalization2D(norm_method=norm_method,
                                        name='norm')(inputs)

    # Add location layer
    if location:
        if frames_per_batch > 1:
            # TODO: TimeDistributed is incompatible with channels_first
            loc = TimeDistributed(Location2D(name='location'),
                                  name='td_location')(norm)
        else:
            # loc = Location2D(name='location')(norm)
            # EDIT: remove name
            # loc = Location2D()(norm)
            t_norm = [norm]
            t_norm.append(Location2D())
            loc = nn.Sequential(*t_norm)
            
        # concat = Concatenate(axis=channel_axis,
        #                      name='concatenate_location')([norm, loc])
        # EDIT: pytorch

        # concat = torch.cat([norm, loc], dim=channel_axis)
        concat = pls_concat_properly(norm, loc)
    else:
        concat = norm

    # Force the channel size for backbone input to be `required_channels`
    # EDIT: 
    # fixed_inputs2 = conv(required_channels, conv_kernel, strides=1,
    #                     padding='same', name='conv_channels')(concat)
    
    # fixed_inputs = conv(concat.shape[channel_axis], required_channels, conv_kernel, stride=1, padding='same')(concat)
    # fixed_inputs = nn.Sequential(concat, conv((inputs.shape[channel_axis]*2), required_channels, conv_kernel, stride=1, padding='same'))
    fixed_inputs = nn.Sequential(concat, conv(required_channels, conv_kernel, stride=1, padding='same'))

    # Force the input shape
    # EDIT:
    # axis = 0 if K.image_data_format() == 'channels_first' else -1
    axis = 0
    
    fixed_input_shape = list(input_shape)
    fixed_input_shape[axis] = required_channels
    fixed_input_shape = tuple(fixed_input_shape)

    model_kwargs = {
        'include_top': False,
        'weights': None,
        'input_shape': fixed_input_shape,
        'pooling': pooling
    }

    _, backbone_dict = get_backbone(backbone, fixed_inputs,
                                    use_imagenet=use_imagenet,
                                    frames_per_batch=frames_per_batch,
                                    return_dict=True,
                                    **model_kwargs)

    backbone_dict_reduced = {k: backbone_dict[k] for k in backbone_dict
                             if k in backbone_levels}
    
    ndim = 2 if frames_per_batch == 1 else 3

    pyramid_dict = create_pyramid_features(backbone_dict_reduced,
                                           ndim=ndim,
                                           lite=lite,
                                           interpolation=interpolation,
                                           upsample_type=upsample_type,
                                           z_axis_convolutions=z_axis_convolutions)

    features = [pyramid_dict[key] for key in pyramid_levels]
    
    if frames_per_batch > 1:
        temporal_features = [__merge_temporal_features(f, mode=temporal_mode,
                                                       frames_per_batch=frames_per_batch)

                             for f in features]
        for f, k in zip(temporal_features, pyramid_levels):
            pyramid_dict[k] = f

    semantic_levels = [int(re.findall(r'\d+', k)[0]) for k in pyramid_dict]
    
    target_level = min(semantic_levels)

    semantic_head_list = []
    if not isinstance(num_semantic_classes, dict):
        num_semantic_classes = {
            k: v for k, v in enumerate(num_semantic_classes)
        }

    for k, v in num_semantic_classes.items():
        semantic_head_list.append(create_semantic_head(
            pyramid_dict, n_classes=v,
            input_target=inputs, target_level=target_level,
            semantic_id=k, ndim=ndim, upsample_type=upsample_type,
            interpolation=interpolation, **kwargs))

    outputs = semantic_head_list
    final_model = outputs
    
    model = combine_models(final_model, device)
    return model