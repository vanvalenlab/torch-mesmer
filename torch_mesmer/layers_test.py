import pytest

import torch

from .layers import Location2D

def help_location_2d(layer_cls, kwargs=None, input_shape=None, input_dtype=None,
               input_data=None, expected_output=None,
               expected_output_dtype=None):
    if input_data is None:
        assert input_shape
        if not input_dtype:
            input_dtype = torch.float
        input_data = 10 * torch.rand(input_shape, dtype=input_dtype)

    assert isinstance(input_data, torch.Tensor)
    
    if input_shape is None:
        input_shape = input_data.shape
    if input_dtype is None:
        input_dtype = input_data.dtype
    
    expected_output_dtype = input_dtype
    expected_output_shape = list(input_shape)
    expected_output_shape[1] = 2
    
    # instantiation
    kwargs = kwargs or {}
    layer = layer_cls(**kwargs)

    # shape and type check
    x = input_data
    y = layer(x)
    assert(list(y.shape) == expected_output_shape)
    assert(y.dtype == expected_output_dtype)

    return y

@pytest.mark.parametrize("input_shape, in_shape, data_format", 
                         [((3, 5, 6, 4), None, "channels_last"), 
                          ((3, 4, 5, 6), (4, 5, 6), "channels_first")])
def test_location_2d(input_shape, in_shape, data_format):
    help_location_2d(
            Location2D,
            kwargs={'in_shape': in_shape,
                    'data_format': data_format},
            input_shape=input_shape)

