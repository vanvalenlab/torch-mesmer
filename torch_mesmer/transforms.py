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

def pixelwise_transform(mask, dilation_radius=None,
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

    channel_axis = 0

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
        numpy.array: the mask squeezed to 2D (or 3D). With ``bins=None`` (the
        default), float32 distances to the background, each object scaled to a
        maximum of 1 when ``normalize`` is set. Otherwise integer distance
        classes from 0 to ``bins - 1``.
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
            normalized_distance = labeled_distance / np.amax(labeled_distance)
            distance[label_matrix == prop.label] = normalized_distance

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

def _distance_per_image(y, distance_transform, mask_dtype, **distance_kwargs):
    """Apply a distance transform to each image of a batch of label masks.

    Args:
        y (numpy.array): labels, ``(batch, 1, *spatial)``.
        distance_transform: ``outer_distance_transform_2d`` or
            ``inner_distance_transform_2d``.
        mask_dtype: dtype of the result.
        distance_kwargs: keyword arguments for ``distance_transform``.

    Returns:
        numpy.array: ``(batch, 1, *spatial)`` of ``mask_dtype``.
    """
    distance = np.zeros((y.shape[0],) + y.shape[2:], dtype=mask_dtype)
    for i in range(y.shape[0]):
        distance[i] = distance_transform(y[i, 0], **distance_kwargs)
    return np.expand_dims(distance, axis=1)


def transform_masks(y, transform, mask_dtype=np.float32, unbatched=False, **kwargs):
    """Turn integer label masks into the training target of one transform.

    Args:
        y (numpy.array): integer labels, channels first, with one channel:
            ``(batch, 1, H, W)`` (or ``(batch, 1, D, H, W)``). With
            ``unbatched=True``, pass ``(1, H, W)`` and a batch axis is added.
        transform (str): one of ``'inner-distance'``, ``'outer-distance'``
            (an underscore works in place of the hyphen), ``'fgbg'`` or
            ``'pixelwise'``. Case is ignored. ``None`` passes the argument
            check but then fails with ``UnboundLocalError``.
        mask_dtype: dtype of the distance transforms' output (default
            float32). ``'fgbg'`` and ``'pixelwise'`` always return uint8.
        unbatched (bool): add a batch axis to ``y`` first.
        kwargs: options for the transform. Those another transform takes are
            ignored.

            * ``'inner-distance'``: ``erosion_width`` (default 0), ``alpha``
              (0.1, or ``'auto'``), ``beta`` (1). See
              ``inner_distance_transform_2d``.
            * ``'outer-distance'``: ``erosion_width`` (default 0). See
              ``outer_distance_transform_2d``.
            * ``'pixelwise'``: ``dilation_radius`` (default ``None``),
              ``separate_edge_classes`` (default ``False``). See
              ``pixelwise_transform``.

    Returns:
        numpy.array: channels first, with the batch axis (also when
        ``unbatched=True``):

        * distance transforms: ``(batch, 1, H, W)``, ``mask_dtype``.
        * ``'fgbg'``: ``(batch, 2, H, W)`` one-hot ``[background, foreground]``.
        * ``'pixelwise'``: ``(batch, 3, H, W)`` one-hot ``[edge, interior,
          background]``, or ``(batch, 4, H, W)`` ``[background edge,
          interior edge, interior, background]`` with
          ``separate_edge_classes=True``.

    Raises:
        ValueError: ``y`` (after ``unbatched``) is not 4- or 5-dimensional.
        ValueError: ``transform`` is not one of the names above.
    """
    valid_transforms = {
        'inner-distance', 
        'inner_distance',
        'outer-distance', 
        'outer_distance',
        'fgbg',
        'pixelwise'
    }

    if unbatched:
        y = np.expand_dims(y, 0)

    if y.ndim not in {4, 5}:
        raise ValueError(f'`labels` data must be of ndim 4 or 5. Got {y.ndim}.')

    if isinstance(transform, str):
        transform = transform.lower()

    if transform not in valid_transforms and transform is not None:
        raise ValueError(f'`{transform}` is not a valid transform')

    elif transform in {'outer-distance', 'outer_distance'}:
        y_transform = _distance_per_image(
            y, outer_distance_transform_2d, mask_dtype,
            erosion_width=kwargs.pop('erosion_width', 0))

    elif transform in {'inner-distance', 'inner_distance'}:
        y_transform = _distance_per_image(
            y, inner_distance_transform_2d, mask_dtype,
            erosion_width=kwargs.pop('erosion_width', 0),
            alpha=kwargs.pop('alpha', 0.1),
            beta=kwargs.pop('beta', 1))

    elif transform == 'fgbg':

        # One-hot [background, foreground], as deepcell's to_categorical gives.
        foreground = (y[:, 0] > 0)
        y_transform = np.stack([~foreground, foreground], axis=1).astype(np.uint8)

    elif transform == 'pixelwise':

        dilation_radius = kwargs.pop('dilation_radius', None)
        separate_edge_classes = kwargs.pop('separate_edge_classes', False)

        edge_class_shape = 4 if separate_edge_classes else 3
        shape = tuple([y.shape[0]] + [edge_class_shape] + list(y.shape[2:]))


        # using uint8 since should only be 4 unique values.
        y_transform = np.zeros(shape, dtype=np.uint8)

        for batch in range(y_transform.shape[0]):
            mask = y[batch, 0, ...]


            y_transform[batch] = pixelwise_transform(
                mask, dilation_radius,
                separate_edge_classes=separate_edge_classes)

    return y_transform
