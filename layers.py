import numpy as np

import torch

from keras.utils.conv_utils import normalize_data_format

'''
class Location2D(Layer):
    """Location Layer for 2D cartesian coordinate locations.

    Args:
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """
    def __init__(self, data_format=None, **kwargs):
        in_shape = kwargs.pop('in_shape', None)
        if in_shape is not None:
            logger.warn('in_shape (from deepcell.layerse.location) is '
                        'deprecated and will be removed in a future version.')
        super().__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        channel_axis = 1 if self.data_format == 'channels_first' else 3
        input_shape[channel_axis] = 2
        return tensor_shape.TensorShape(input_shape)

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            x = K.arange(0, input_shape[2], dtype=inputs.dtype)
            y = K.arange(0, input_shape[3], dtype=inputs.dtype)
        else:
            x = K.arange(0, input_shape[1], dtype=inputs.dtype)
            y = K.arange(0, input_shape[2], dtype=inputs.dtype)

        x = x / K.max(x)
        y = y / K.max(y)

        loc_x, loc_y = tf.meshgrid(x, y, indexing='ij')

        if self.data_format == 'channels_first':
            loc = K.stack([loc_x, loc_y], axis=0)
        else:
            loc = K.stack([loc_x, loc_y], axis=-1)

        location = K.expand_dims(loc, axis=0)
        if self.data_format == 'channels_first':
            location = K.permute_dimensions(location, pattern=[0, 2, 3, 1])

        location = tf.tile(location, [input_shape[0], 1, 1, 1])

        if self.data_format == 'channels_first':
            location = K.permute_dimensions(location, pattern=[0, 3, 1, 2])

        return location

    def get_config(self):
        config = {
            'data_format': self.data_format
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
'''


# class Attention_module(torch.nn.Module):
#     def __init__(self, class_num, input_shape):
#         super().__init__()
#         self.class_num = class_num
#         embedding_length = int(input_shape[2])
#         self.Ws = torch.nn.Embedding(num_embeddings=class_num, 
#                                      embedding_dim=embedding_length)  # Embedding layer
#         torch.nn.init.xavier_uniform_(self.Ws.weight) # Glorot initialization

class Location2D(torch.nn.Module):
    """Location Layer for 2D cartesian coordinate locations.

    Args:
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """
    def __init__(self, data_format=None, **kwargs):
        in_shape = kwargs.pop('in_shape', None)
        if in_shape is not None:
            logger.warn('in_shape (from deepcell.layerse.location) is '
                        'deprecated and will be removed in a future version.')
        super().__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.data_format = "channels_first"
        print(self.data_format)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        channel_axis = 1 if self.data_format == 'channels_first' else 3
        channel_axis = 1
        input_shape[channel_axis] = 2
        return tensor_shape.TensorShape(input_shape)

    def forward(self, inputs):
        input_shape = inputs.size()
        input_device = inputs.device
            
        if self.data_format == 'channels_first':
            # print("INPUT_SHAPE", input_shape[2])
            # print("DTYPE", inputs.dtype)
            x = torch.arange(0, input_shape[2], dtype=inputs.dtype)
            y = torch.arange(0, input_shape[3], dtype=inputs.dtype)
        else:
            assert(False)
            x = torch.arange(0, input_shape[1], dtype=inputs.dtype)
            y = torch.arange(0, input_shape[2], dtype=inputs.dtype)

        x = x / torch.max(x)
        y = y / torch.max(y)

        loc_x, loc_y = torch.meshgrid(x, y, indexing='ij')

        if self.data_format == 'channels_first':
            loc = torch.stack([loc_x, loc_y], dim=0)
        else:
            assert(False)
            loc = torch.stack([loc_x, loc_y], dim=-1)

        location = torch.unsqueeze(loc, dim=0)
        if self.data_format == 'channels_first':
            location = torch.permute(location, dims=[0, 2, 3, 1])

        location = torch.tile(location, [input_shape[0], 1, 1, 1])

        if self.data_format == 'channels_first':
            location = torch.permute(location, dims=[0, 3, 1, 2])

        return location.to(input_device)

    def get_config(self):
        config = {
            'data_format': self.data_format
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))