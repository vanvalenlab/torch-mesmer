import numpy as np

import scipy.ndimage as nd
import skimage

eps = 1e-7

def erode_edges(mask, erosion_width):
    """Erode edge of objects to prevent them from touching

    Args:
        mask (numpy.array): uniquely labeled instance mask
        erosion_width (int): integer value for pixel width to erode edges

    Returns:
        numpy.array: mask where each instance has had the edges eroded

    Raises:
        ValueError: mask.ndim is not 2 or 3
    """

    if mask.ndim not in {2, 3}:
        raise ValueError('erode_edges expects arrays of ndim 2 or 3.'
                         'Got ndim: {}'.format(mask.ndim))
    if erosion_width:
        new_mask = np.copy(mask)
        for _ in range(erosion_width):
            boundaries = skimage.segmentation.find_boundaries(new_mask, mode='inner')
            new_mask[boundaries > 0] = 0
        return new_mask

    return mask

def pixelwise_transform(mask, dilation_radius=None, data_format=None,
                        separate_edge_classes=False):
    """Transforms a label mask for a z stack edge, interior, and background

    Args:
        mask (numpy.array): tensor of labels
        dilation_radius (int):  width to enlarge the edge feature of
            each instance
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        separate_edge_classes (bool): Whether to separate the cell edge class
            into 2 distinct cell-cell edge and cell-background edge classes.

    Returns:
        numpy.array: An array with the same shape as ``mask``, except the
        channel axis will be a one-hot encoded semantic segmentation for
        3 main features:
        ``[cell_edge, cell_interior, background]``.
        If ``separate_edge_classes`` is ``True``, the ``cell_interior``
        feature is split into 2 features and the resulting channels are:
        ``[bg_cell_edge, cell_cell_edge, cell_interior, background]``.
    """

    assert data_format is not None, print("Please provide a data format")

    if data_format == 'channels_first':
        channel_axis = 0
    else:
        channel_axis = -1

    # Detect the edges and interiors
    edge = skimage.segmentation.find_boundaries(mask, mode='inner').astype('int')
    interior = np.logical_and(edge == 0, mask > 0).astype('int')

    if not separate_edge_classes:
        if dilation_radius:
            dil_strel = skimage.morphology.ball(dilation_radius) if mask.ndim > 2 else skimage.morphology.disk(dilation_radius)
            # Thicken cell edges to be more pronounced
            edge = skimage.morphology.dilation(edge, footprint=dil_strel)

            # Thin the augmented edges by subtracting the interior features.
            edge = (edge - interior > 0).astype('int')

        background = (1 - edge - interior > 0)
        background = background.astype('int')

        all_stacks = [
            edge,
            interior,
            background
        ]

        return np.stack(all_stacks, axis=channel_axis)
    
    else:
        strel = skimage.morphology.ball(1) if mask.ndim > 2 else skimage.morphology.disk(1)
        # dilate the background masks and subtract from all edges for background-edges
        background = (mask == 0).astype('int')
        dilated_background = skimage.morphology.binary_dilation(background, strel)

        background_edge = (edge - dilated_background > 0).astype('int')

        # edges that are not background-edges are interior-edges
        interior_edge = (edge - background_edge > 0).astype('int')

        if dilation_radius:
            dil_strel = skimage.morphology.ball(dilation_radius) if mask.ndim > 2 else skimage.morphology.disk(dilation_radius)
            # Thicken cell edges to be more pronounced
            interior_edge = skimage.morphology.binary_dilation(interior_edge, footprint=dil_strel)
            background_edge = skimage.morphology.binary_dilation(background_edge, footprint=dil_strel)

            # Thin the augmented edges by subtracting the interior features.
            interior_edge = (interior_edge - interior > 0).astype('int')
            background_edge = (background_edge - interior > 0).astype('int')

        background = (1 - background_edge - interior_edge - interior > 0)
        background = background.astype('int')

        all_stacks = [
            background_edge,
            interior_edge,
            interior,
            background
        ]

        return np.stack(all_stacks, axis=channel_axis)


def outer_distance_transform_2d(mask, bins=None, erosion_width=None,
                                normalize=True):
    """Transform a label mask with an outer distance transform.

    Args:
        mask (numpy.array): A label mask (``y`` data).
        bins (int): The number of transformed distance classes. If ``None``,
            returns the continuous outer transform.
        erosion_width (int): Number of pixels to erode edges of each labels
        normalize (bool): Normalize the transform of each cell by that
            cell's largest distance.

    Returns:
        numpy.array: A mask of same shape as input mask,
        with each label being a distance class from 1 to ``bins``.
    """
    mask = np.squeeze(mask)  # squeeze the channels
    mask = erode_edges(mask, erosion_width)

    distance = nd.distance_transform_edt(mask)
    distance = distance.astype(np.float32)  # normalized distances are floats

    if normalize:
        # uniquely label each cell and normalize the distance values
        # by that cells maximum distance value
        label_matrix = skimage.measure.label(mask)
        for prop in skimage.measure.regionprops(label_matrix):
            labeled_distance = distance[label_matrix == prop.label]
            normalized_distance = 1 - labeled_distance / np.amax(labeled_distance)
            distance[label_matrix == prop.label] = normalized_distance

    distance[mask == 0] = 0

    if bins is None:
        return distance

    # bin each distance value into a class from 1 to bins
    min_dist = np.amin(distance)
    max_dist = np.amax(distance)
    distance_bins = np.linspace(min_dist - eps,
                                max_dist + eps,
                                num=bins + 1)
    distance = np.digitize(distance, distance_bins, right=True)
    return distance - 1  # minimum distance should be 0, not 1


def outer_distance_transform_3d(mask, bins=None, erosion_width=None,
                                normalize=True, sampling=[0.5, 0.217, 0.217]):
    """Transforms a label mask for a z stack with an outer distance transform.
    Uses scipy's distance_transform_edt

    Args:
        mask (numpy.array): A z-stack of label masks (``y`` data).
        bins (int): The number of transformed distance classes.
        erosion_width (int): Number of pixels to erode edges of each labels.
        normalize (bool): Normalize the transform of each cell by that
            cell's largest distance.
        sampling (list): Spacing of pixels along each dimension.

    Returns:
        numpy.array: 3D Euclidiean Distance Transform
    """
    maskstack = np.squeeze(mask)  # squeeze the channels
    maskstack = erode_edges(maskstack, erosion_width)

    distance = nd.distance_transform_edt(maskstack, sampling=sampling)

    # normalize by maximum distance
    if normalize:
        for cell_label in np.unique(maskstack):
            if cell_label == 0:  # distance is only found for non-zero regions
                continue
            index = np.nonzero(maskstack == cell_label)
            distance[index] = distance[index] / np.amax(distance[index])

    if bins is None:
        return distance

    # divide into bins
    min_dist = np.amin(distance.flatten())
    max_dist = np.amax(distance.flatten())
    distance_bins = np.linspace(min_dist - eps,
                                max_dist + eps,
                                num=bins + 1)
    distance = np.digitize(distance, distance_bins, right=True)
    return distance - 1  # minimum distance should be 0, not 1


def outer_distance_transform_movie(mask, bins=None, erosion_width=None,
                                   normalize=True):
    """Transform a label mask for a movie with an outer distance transform.
    Applies the 2D transform to each frame.

    Args:
        mask (numpy.array): A label mask (``y`` data).
        bins (int): The number of transformed distance classes.
        erosion_width (int): number of pixels to erode edges of each labels.
        normalize (bool): Normalize the transform of each cell by that
            cell's largest distance.

    Returns:
        numpy.array: a mask of same shape as input mask,
        with each label being a distance class from 1 to ``bins``
    """
    distances = []
    for frame in range(mask.shape[0]):
        mask_frame = mask[frame]

        distance = outer_distance_transform_2d(
            mask_frame, bins=bins,
            erosion_width=erosion_width,
            normalize=normalize)

        distances.append(distance)

    distances = np.stack(distances, axis=0)

    return distances


def inner_distance_transform_2d(mask, bins=None, erosion_width=None,
                                alpha=0.1, beta=1):
    """Transform a label mask with an inner distance transform.

    .. code-block:: python

        inner_distance = 1 / (1 + beta * alpha * distance_to_center)

    Args:
        mask (numpy.array): A label mask (``y`` data).
        bins (int): The number of transformed distance classes.
        erosion_width (int): number of pixels to erode edges of each labels
        alpha (float, str): coefficent to reduce the magnitude of the distance
            value. If "auto", determines ``alpha`` for each cell based on the
            cell area.
        beta (float): scale parameter that is used when ``alpha`` is "auto".

    Returns:
        numpy.array: a mask of same shape as input mask,
        with each label being a distance class from 1 to ``bins``.

    Raises:
        ValueError: ``alpha`` is a string but not set to "auto".
    """
    # Check input to alpha
    if isinstance(alpha, str):
        if alpha.lower() != 'auto':
            raise ValueError('alpha must be set to "auto"')

    mask = np.squeeze(mask)
    mask = erode_edges(mask, erosion_width)

    distance = nd.distance_transform_edt(mask)
    distance = distance.astype(np.float32)

    label_matrix = skimage.measure.label(mask)

    inner_distance = np.zeros(distance.shape, dtype=np.float32)
    for prop in skimage.measure.regionprops(label_matrix, distance):
        coords = prop.coords
        center = prop.centroid_weighted
        distance_to_center = np.sum((coords - center) ** 2, axis=1)

        # Determine alpha to use
        if str(alpha).lower() == 'auto':
            _alpha = 1 / np.sqrt(prop.area)
        else:
            _alpha = float(alpha)

        center_transform = 1 / (1 + beta * _alpha * distance_to_center)
        coords_x = coords[:, 0]
        coords_y = coords[:, 1]
        inner_distance[coords_x, coords_y] = center_transform
        
    distance[mask == 0] = 0

    if bins is None:
        return inner_distance

    # divide into bins
    min_dist = np.amin(inner_distance.flatten())
    max_dist = np.amax(inner_distance.flatten())
    distance_bins = np.linspace(min_dist - eps,
                                max_dist + eps,
                                num=bins + 1)
    inner_distance = np.digitize(inner_distance, distance_bins, right=True)
    return inner_distance - 1  # minimum distance should be 0, not 1


def inner_distance_transform_3d(mask, bins=None,
                                erosion_width=None,
                                alpha=0.1, beta=1,
                                sampling=[0.5, 0.217, 0.217]):
    """Transform a label mask for a z-stack with an inner distance transform.

    .. code-block:: python

        inner_distance = 1 / (1 + beta * alpha * distance_to_center)

    Args:
        mask (numpy.array): A label mask (``y`` data).
        bins (int): The number of transformed distance classes.
        erosion_width (int): Number of pixels to erode edges of each labels
        alpha (float, str): Coefficent to reduce the magnitude of the distance
            value. If ``'auto'``, determines alpha for each cell based on the
            cell area.
        beta (float): Scale parameter that is used when ``alpha`` is "auto".
        sampling (list): Spacing of pixels along each dimension.

    Returns:
        numpy.array: A mask of same shape as input mask,
        with each label being a distance class from 1 to ``bins``.

    Raises:
        ValueError: ``alpha`` is a string but not set to "auto".
    """
    # Check input to alpha
    if isinstance(alpha, str):
        if alpha.lower() != 'auto':
            raise ValueError('alpha must be set to "auto"')

    mask = np.squeeze(mask)
    mask = erode_edges(mask, erosion_width)

    distance = nd.distance_transform_edt(mask, sampling=sampling)
    distance = distance.astype(np.float32)

    label_matrix = skimage.measure.label(mask)

    inner_distance = np.zeros(distance.shape, dtype=np.float32)
    for prop in skimage.measure.regionprops(label_matrix, distance):
        coords = prop.coords
        center = prop.weighted_centroid
        distance_to_center = (coords - center) * np.array(sampling)
        distance_to_center = np.sum(distance_to_center ** 2, axis=1)

        # Determine alpha to use
        if str(alpha).lower() == 'auto':
            _alpha = 1 / np.cbrt(prop.area)
        else:
            _alpha = float(alpha)

        center_transform = 1 / (1 + beta * _alpha * distance_to_center)
        coords_z = coords[:, 0]
        coords_x = coords[:, 1]
        coords_y = coords[:, 2]
        inner_distance[coords_z, coords_x, coords_y] = center_transform

    if bins is None:
        return inner_distance

    # divide into bins
    min_dist = np.amin(inner_distance.flatten())
    max_dist = np.amax(inner_distance.flatten())
    distance_bins = np.linspace(min_dist - eps,
                                max_dist + eps,
                                num=bins + 1)
    inner_distance = np.digitize(inner_distance, distance_bins, right=True)
    return inner_distance - 1  # minimum distance should be 0, not 1


def inner_distance_transform_movie(mask, bins=None, erosion_width=None,
                                   alpha=0.1, beta=1):
    """Transform a label mask with an inner distance transform. Applies the
    2D transform to each frame.

    Args:
        mask (numpy.array): A label mask (``y`` data).
        bins (int): The number of transformed distance classes.
        erosion_width (int): Number of pixels to erode edges of each labels.
        alpha (float, str): Coefficent to reduce the magnitude of the distance
            value. If "auto", determines ``alpha`` for each cell based on the
            cell area.
        beta (float): Scale parameter that is used when ``alpha`` is "auto".

    Returns:
        numpy.array: A mask of same shape as input mask,
        with each label being a distance class from 1 to ``bins``.

    Raises:
        ValueError: ``alpha`` is a string but not set to "auto".
    """
    # Check input to alpha
    if isinstance(alpha, str):
        if alpha.lower() != 'auto':
            raise ValueError('alpha must be set to "auto"')

    inner_distances = []

    for frame in range(mask.shape[0]):
        mask_frame = mask[frame]

        inner_distance = inner_distance_transform_2d(
            mask_frame, bins=bins,
            erosion_width=erosion_width,
            alpha=alpha, beta=beta)

        inner_distances.append(inner_distance)

    inner_distances = np.stack(inner_distances, axis=0)

    return inner_distances

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

def transform_masks(y, transform, data_format=None, mask_dtype=np.float32, unbatched=False, **kwargs):
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
        numpy.array: the output of the given transform function on ``y``. (with batch dimension)

    Raises:
        ValueError: Rank of ``y`` is not 4 or 5.
        ValueError: Channel dimension of ``y`` is not 1.
        ValueError: ``transform`` is invalid value.
    """
    valid_transforms = {
        'inner-distance', 
        'inner_distance',
        'outer-distance', 
        'outer_distance',
        'fgbg',
        'pixelwise'
    }
    
    if data_format is None:
        data_format = "channels_last"

    if unbatched:
        y = np.expand_dims(y, 0)

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

    elif transform in {'outer-distance', 'outer_distance'}:

        distance_kwargs = {
            'erosion_width': kwargs.pop('erosion_width', 0),
        }

        if data_format == 'channels_first':
            shape = tuple([y.shape[0]] + list(y.shape[2:]))
        else:
            shape = y.shape[0:-1]

        y_transform = np.zeros(shape, dtype=mask_dtype)

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]
            y_transform[batch] = outer_distance_transform_2d(mask, **distance_kwargs)

        y_transform = np.expand_dims(y_transform, axis=-1)

        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform in {'inner-distance', 'inner_distance'}:

        distance_kwargs = {
            'erosion_width': kwargs.pop('erosion_width', 0),
            'alpha': kwargs.pop('alpha', 0.1),
            'beta': kwargs.pop('beta', 1)
        }

        if data_format == 'channels_first':
            shape = tuple([y.shape[0]] + list(y.shape[2:]))
        else:
            shape = y.shape[0:-1]

        y_transform = np.zeros(shape, dtype=mask_dtype)

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]
            y_transform[batch] = inner_distance_transform_2d(mask, **distance_kwargs)

        y_transform = np.expand_dims(y_transform, axis=-1)

        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform == 'fgbg':

        y_transform = np.where(y > 1, 1, y)
        
        # y_transform = nd.binary_erosion(y_transform, structure=np.ones((1,1,2,2))).astype(y_transform.dtype)

        # # convert to one hot notation
        # if data_format == 'channels_first':
        #     y_transform = np.rollaxis(y_transform, 1, y.ndim)
        
        # # using uint8 since should only be 2 unique values.
        # y_transform = to_categorical(y_transform, dtype=np.uint8)

        # if data_format == 'channels_first':
        #     y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform == 'pixelwise':

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

    return y_transform