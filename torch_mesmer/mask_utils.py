import numpy as np
import warnings

from .transform_utils import pixelwise_transform
from .transform_utils import outer_distance_transform_movie, outer_distance_transform_3d, outer_distance_transform_2d
from .transform_utils import inner_distance_transform_movie, inner_distance_transform_3d, inner_distance_transform_2d

# Copied from keras
def to_categorical(x, num_classes=None, dtype="int64"):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with `categorical_crossentropy`.

    Args:
        x: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
            as `max(x) + 1`. Defaults to `None`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.

    Example:

    >>> a = to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    >>> b = np.array([.9, .04, .03, .03,
    ...               .3, .45, .15, .13,
    ...               .04, .01, .94, .05,
    ...               .12, .21, .5, .17],
    ...               shape=[4, 4])
    >>> loss = keras.ops.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]

    >>> loss = keras.ops.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]
    """
    x = np.array(x, dtype="int64")
    input_shape = x.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    x = x.reshape(-1)
    if not num_classes:
        num_classes = np.max(x) + 1
    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, num_classes))
    categorical[np.arange(batch_size), x] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical.astype(dtype)


def _transform_masks(y, transform, data_format=None, mask_dtype=np.float32, **kwargs):
    """Based on the transform key, apply a transform function to the masks.

    Refer to :mod:`torch_mesmer.transform_utils` for more information about
    available transforms. Caution for unknown transform keys.

    Args:
        y (numpy.array): Labels of ``ndim`` 4 or 5
        transform (str): Name of the transform, one of
            ``{"deepcell", "disc", "watershed", None}``.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        kwargs (dict): Optional transform keyword arguments.

    Returns:
        numpy.array: the output of the given transform function on ``y``.

    Raises:
        ValueError: Rank of ``y`` is not 4 or 5.
        ValueError: Channel dimension of ``y`` is not 1.
        ValueError: ``transform`` is invalid value.
    """
    valid_transforms = {
        'deepcell',  # deprecated for "pixelwise"
        'pixelwise',
        'disc',
        'watershed',  # deprecated for "outer-distance"
        'watershed-cont',  # deprecated for "outer-distance"
        'inner-distance', 'inner_distance',
        'outer-distance', 'outer_distance',
        'centroid',  # deprecated for "inner-distance"
        'fgbg'
    }
    if data_format is None:
        data_format = "channels_last"

    if y.ndim not in {4, 5}:
        raise ValueError('`labels` data must be of ndim 4 or 5.  Got', y.ndim)

    channel_axis = 1 if data_format == 'channels_first' else -1

    if y.shape[channel_axis] != 1:
        raise ValueError('Expected channel axis to be 1 dimension. Got',
                         y.shape[1 if data_format == 'channels_first' else -1])

    if isinstance(transform, str):
        transform = transform.lower()

    if transform not in valid_transforms and transform is not None:
        raise ValueError(f'`{transform}` is not a valid transform')

    if transform in {'pixelwise', 'deepcell'}:
        if transform == 'deepcell':
            warnings.warn(f'The `{transform}` transform is deprecated. Please use the '
                          '`pixelwise` transform instead.',
                          DeprecationWarning)
        dilation_radius = kwargs.pop('dilation_radius', None)
        separate_edge_classes = kwargs.pop('separate_edge_classes', False)

        edge_class_shape = 4 if separate_edge_classes else 3

        if data_format == 'channels_first':
            shape = tuple([y.shape[0]] + [edge_class_shape] + list(y.shape[2:]))
        else:
            shape = tuple(list(y.shape[0:-1]) + [edge_class_shape])

        # using uint8 since should only be 4 unique values.
        y_transform = np.zeros(shape, dtype=np.uint8)

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]

            y_transform[batch] = pixelwise_transform(
                mask, dilation_radius, data_format=data_format,
                separate_edge_classes=separate_edge_classes)

    elif transform in {'outer-distance', 'outer_distance',
                       'watershed', 'watershed-cont'}:
        if transform in {'watershed', 'watershed-cont'}:
            warnings.warn(f'The `{transform}` transform is deprecated. Please use the '
                          '`outer-distance` transform instead.',
                          DeprecationWarning)

        by_frame = kwargs.pop('by_frame', True)
        bins = kwargs.pop('distance_bins', None)

        distance_kwargs = {
            'bins': bins,
            'erosion_width': kwargs.pop('erosion_width', 0),
        }

        # If using 3d transform, pass in scale arg
        if y.ndim == 5 and not by_frame:
            distance_kwargs['sampling'] = kwargs.pop('sampling', [0.5, 0.217, 0.217])

        if data_format == 'channels_first':
            shape = tuple([y.shape[0]] + list(y.shape[2:]))
        else:
            shape = y.shape[0:-1]
        y_transform = np.zeros(shape, dtype=mask_dtype)

        if y.ndim == 5:
            if by_frame:
                _distance_transform = outer_distance_transform_movie
            else:
                _distance_transform = outer_distance_transform_3d
        else:
            _distance_transform = outer_distance_transform_2d

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]
            y_transform[batch] = _distance_transform(mask, **distance_kwargs)

        y_transform = np.expand_dims(y_transform, axis=-1)

        if bins is not None:
            # convert to one hot notation
            # uint8's max value of255 seems like a generous limit for binning.
            y_transform = to_categorical(y_transform, num_classes=bins, dtype=np.uint8)
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform in {'inner-distance', 'inner_distance', 'centroid'}:
        if transform == 'centroid':
            warnings.warn(f'The `{transform}` transform is deprecated. Please use the '
                          '`inner-distance` transform instead.',
                          DeprecationWarning)

        by_frame = kwargs.pop('by_frame', True)
        bins = kwargs.pop('distance_bins', None)

        distance_kwargs = {
            'bins': bins,
            'erosion_width': kwargs.pop('erosion_width', 0),
            'alpha': kwargs.pop('alpha', 0.1),
            'beta': kwargs.pop('beta', 1)
        }

        # If using 3d transform, pass in scale arg
        if y.ndim == 5 and not by_frame:
            distance_kwargs['sampling'] = kwargs.pop('sampling', [0.5, 0.217, 0.217])

        if data_format == 'channels_first':
            shape = tuple([y.shape[0]] + list(y.shape[2:]))
        else:
            shape = y.shape[0:-1]
        y_transform = np.zeros(shape, dtype=mask_dtype)

        if y.ndim == 5:
            if by_frame:
                _distance_transform = inner_distance_transform_movie
            else:
                _distance_transform = inner_distance_transform_3d
        else:
            _distance_transform = inner_distance_transform_2d

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]
            y_transform[batch] = _distance_transform(mask, **distance_kwargs)

        y_transform = np.expand_dims(y_transform, axis=-1)

        if distance_kwargs['bins'] is not None:
            # convert to one hot notation
            # uint8's max value of255 seems like a generous limit for binning.
            y_transform = to_categorical(y_transform, num_classes=bins, dtype=np.uint8)
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform == 'disc' or transform is None:
        dtype = mask_dtype if transform == 'disc' else np.int32
        y_transform = to_categorical(y.squeeze(channel_axis), dtype=dtype)
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform == 'fgbg':
        y_transform = np.where(y > 1, 1, y)
        # convert to one hot notation
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, 1, y.ndim)
        # using uint8 since should only be 2 unique values.
        y_transform = to_categorical(y_transform, dtype=np.uint8)
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    return y_transform
