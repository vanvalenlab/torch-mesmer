import torch
import warnings

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
            warnings.warn('in_shape is deprecated and will be removed in a future version.')
        super().__init__(**kwargs)
        self.data_format = "channels_first"

    def forward(self, inputs):
        input_shape = inputs.size()
        input_device = inputs.device
            
        if self.data_format == 'channels_first':
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