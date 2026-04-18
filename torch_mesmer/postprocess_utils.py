import numpy as np
import scipy.ndimage as nd

from scipy.signal import windows
from torch_mesmer.utils import histogram_normalization, percentile_threshold, resize, erode_edges, fill_holes

from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import remove_small_objects, h_maxima
from skimage.morphology import disk, square, dilation
from skimage.segmentation import relabel_sequential, watershed

def spline_window(window_size, overlap_left, overlap_right, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """

    def _spline_window(w_size):
        intersection = int(w_size / 4)
        wind_outer = (abs(2 * (windows.triang(w_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (windows.triang(w_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.amax(wind)
        return wind

    # Create the window for the left overlap
    if overlap_left > 0:
        window_size_l = 2 * overlap_left
        l_spline = _spline_window(window_size_l)[0:overlap_left]

    # Create the window for the right overlap
    if overlap_right > 0:
        window_size_r = 2 * overlap_right
        r_spline = _spline_window(window_size_r)[overlap_right:]

    # Put the two together
    window = np.ones((window_size,))
    if overlap_left > 0:
        window[0:overlap_left] = l_spline
    if overlap_right > 0:
        window[-overlap_right:] = r_spline

    return window

def window_2D(window_size, overlap_x=(32, 32), overlap_y=(32, 32), power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Returns a channels-first compatible window of shape (1, tile_x, tile_y).
    """
    window_x = spline_window(window_size[0], overlap_x[0], overlap_x[1], power=power)
    window_y = spline_window(window_size[1], overlap_y[0], overlap_y[1], power=power)

    # Reshape for channels-first broadcasting: (tile_x, 1) * (1, tile_y) -> (tile_x, tile_y)
    window = window_x[:, np.newaxis] * window_y[np.newaxis, :]

    # Add channel dim at front: (1, tile_x, tile_y) for broadcasting over (C, tile_x, tile_y)
    return window[np.newaxis, :, :]


def untile_image(tiles, tiles_info, power=2, **kwargs):
    """Untile a set of tiled images back to the original model shape.
    Expects channels-first format: (B, C, H, W).

    Args:
        tiles (numpy.array): The tiled images to untile, shape (N, C, tile_x, tile_y).
        tiles_info (dict): Details of how the image was tiled (from tile_image).
        power (int): The power of the window function.

    Returns:
        numpy.array: The untiled image, shape (B, C, H, W).
    """
    min_tile_size = 32
    min_stride_ratio = 0.5

    stride_ratio = tiles_info['stride_ratio']
    image_shape = tiles_info['image_shape']
    tile_size_x = tiles_info['tile_size_x']
    tile_size_y = tiles_info['tile_size_y']
    x_pad = tiles_info['pad_x']
    y_pad = tiles_info['pad_y']

    # Channels-first: (B, C, H, W) — use tiles.shape[1] for n_channels
    image_shape = (image_shape[0], tiles.shape[1], image_shape[2], image_shape[3])
    image = np.zeros(image_shape, dtype=float)

    # Precompute window cache keyed on overlap pairs
    window_cache = {}
    for overlap_x, overlap_y in zip(tiles_info['overlaps_x'], tiles_info['overlaps_y']):
        key = (overlap_x, overlap_y)
        if key not in window_cache:
            window_cache[key] = window_2D(
                (tile_size_x, tile_size_y),
                overlap_x=overlap_x,
                overlap_y=overlap_y,
                power=power
            )

    use_spline = (
        min_tile_size <= tile_size_x < image_shape[2] and
        min_tile_size <= tile_size_y < image_shape[3] and
        stride_ratio >= min_stride_ratio
    )

    for tile, batch, x_start, x_end, y_start, y_end, overlap_x, overlap_y in zip(
            tiles,
            tiles_info['batches'],
            tiles_info['x_starts'], tiles_info['x_ends'],
            tiles_info['y_starts'], tiles_info['y_ends'],
            tiles_info['overlaps_x'], tiles_info['overlaps_y']):

        if use_spline:
            window = window_cache[(overlap_x, overlap_y)]
            # Channels-first slice: (C, tile_x, tile_y) * (1, tile_x, tile_y)
            image[batch, :, x_start:x_end, y_start:y_end] += tile * window
        else:
            image[batch, :, x_start:x_end, y_start:y_end] = tile

    image = image.astype(tiles.dtype)

    # Unpad spatial dims (axes 2 and 3)
    x_end = image_shape[2] - x_pad[1] if x_pad[1] != 0 else None
    y_end = image_shape[3] - y_pad[1] if y_pad[1] != 0 else None

    return image[:, :, x_pad[0]:x_end, y_pad[0]:y_end]


def tile_image(image, model_input_shape=(512, 512), stride_ratio=1.0, pad_mode='constant'):
    """
    Tile large image into overlapping tiles of size `model_input_shape`.
    Expects and returns channels-first format: (B, C, H, W).

    Args:
        image (numpy.array): The image to tile, must be rank 4 (B, C, H, W).
        model_input_shape (tuple): The (x, y) spatial input size of the model.
        stride_ratio (float): Stride as a fraction of tile size.
        pad_mode (str): Padding mode passed to ``np.pad``.

    Returns:
        tuple: (numpy.array, dict): Tiled images array (B, C, tile_x, tile_y)
            and tiling metadata dict.

    Raises:
        ValueError: image is not rank 4.
    """
    if image.ndim != 4:
        raise ValueError('Expected image of rank 4, got {}'.format(image.ndim))

    tile_size_x, tile_size_y = model_input_shape
    # Channels-first: spatial dims are axes 2 and 3
    image_size_x, image_size_y = image.shape[2], image.shape[3]

    def even_stride(ratio, tile_size):
        return min(int(np.ceil(ratio * tile_size / 2.0) * 2), tile_size)

    stride_x = even_stride(stride_ratio, tile_size_x)
    stride_y = even_stride(stride_ratio, tile_size_y)

    rep_x = max(int(np.ceil((image_size_x - tile_size_x) / stride_x + 1)), 1)
    rep_y = max(int(np.ceil((image_size_y - tile_size_y) / stride_y + 1)), 1)

    def edge_padding(overlap):
        return (int(np.ceil(overlap / 2)), int(np.floor(overlap / 2)))

    overlap_x = tile_size_x + stride_x * (rep_x - 1) - image_size_x
    overlap_y = tile_size_y + stride_y * (rep_y - 1) - image_size_y
    pad_x = edge_padding(overlap_x)
    pad_y = edge_padding(overlap_y)

    # Channels-first padding: (B, C, H, W) → pad axes 2 and 3, not 1 and 2
    image = np.pad(image, [(0, 0), (0, 0), pad_x, pad_y], pad_mode)
    img_x, img_y = image.shape[2], image.shape[3]

    def tile_indices(rep, stride, tile_size, img_size):
        starts = [i * stride if i < rep - 1 else img_size - tile_size for i in range(rep)]
        ends = [s + tile_size for s in starts]
        return starts, ends

    x_starts, x_ends = tile_indices(rep_x, stride_x, tile_size_x, img_x)
    y_starts, y_ends = tile_indices(rep_y, stride_y, tile_size_y, img_y)

    def compute_overlaps(rep, stride, tile_size, starts, img_size):
        overlaps = []
        for i in range(rep):
            if i == 0:
                ov = (0, tile_size - stride)
            elif i == rep - 2:
                ov = (tile_size - stride, tile_size - img_size + starts[i] + tile_size)
            elif i == rep - 1:
                ov = (starts[i - 1] + tile_size - starts[i], 0) if rep > 1 else (0, 0)
            else:
                ov = (tile_size - stride, tile_size - stride)
            overlaps.append(ov)
        return overlaps

    overlaps_x = compute_overlaps(rep_x, stride_x, tile_size_x, x_starts, img_x)
    overlaps_y = compute_overlaps(rep_y, stride_y, tile_size_y, y_starts, img_y)

    batch_size, n_channels = image.shape[0], image.shape[1]
    n_tiles = batch_size * rep_x * rep_y

    # Channels-first tile shape: (N, C, tile_x, tile_y)
    tiles = np.zeros((n_tiles, n_channels, tile_size_x, tile_size_y), dtype=image.dtype)

    indices = [(b, i, j)
               for b in range(batch_size)
               for i in range(rep_x)
               for j in range(rep_y)]

    for counter, (b, i, j) in enumerate(indices):
        # Slice spatial dims 2 and 3, keeping all channels (dim 1)
        tiles[counter] = image[b, :, x_starts[i]:x_ends[i], y_starts[j]:y_ends[j]]

    flat_b, flat_i, flat_j = zip(*indices)

    return tiles, {
        'batches':      list(flat_b),
        'x_starts':     [x_starts[i] for i in flat_i],
        'x_ends':       [x_ends[i]   for i in flat_i],
        'y_starts':     [y_starts[j] for j in flat_j],
        'y_ends':       [y_ends[j]   for j in flat_j],
        'overlaps_x':   [overlaps_x[i] for i in flat_i],
        'overlaps_y':   [overlaps_y[j] for j in flat_j],
        'stride_x':     stride_x,
        'stride_y':     stride_y,
        'tile_size_x':  tile_size_x,
        'tile_size_y':  tile_size_y,
        'stride_ratio': stride_ratio,
        'image_shape':  image.shape,
        'dtype':        image.dtype,
        'pad_x':        pad_x,
        'pad_y':        pad_y,
    }

# pre- and post-processing functions
def mesmer_preprocess(image, **kwargs):
    """Preprocess input data for Mesmer model.

    Args:
        image: array to be processed

    Returns:
        np.array: processed image array
    """

    data_format = kwargs.get('data_format', 'channels_first')
    if data_format == 'channels_first':
        image = np.moveaxis(image, 1, -1)

    if len(image.shape) != 4:
        raise ValueError(f"Image data must be 4D, got image of shape {image.shape}")

    output = np.copy(image)
    threshold = kwargs.get('threshold', True)
    if threshold:
        percentile = kwargs.get('percentile', 99.9)
        output = percentile_threshold(image=output, percentile=percentile)

    normalize = kwargs.get('normalize', True)
    if normalize:
        kernel_size = kwargs.get('kernel_size', 128)
        output = histogram_normalization(image=output, kernel_size=kernel_size)

    if data_format == 'channels_first':
        output = np.moveaxis(output, -1, 1)
    
    return output


def format_output_mesmer(output_list):
    """Takes list of model outputs and formats into a dictionary for better readability

    Args:
        output_list (list): predictions from semantic heads

    Returns:
        dict: Dict of predictions for whole cell and nuclear.

    Raises:
        ValueError: if model output list is not len(4)
    """
    expected_length = 4
    if len(output_list) != expected_length:
        raise ValueError('output_list was length {}, expecting length {}'.format(
            len(output_list), expected_length))

    formatted_dict = {
        'whole-cell': [output_list[0], output_list[1][..., 1:2]],
        'nuclear': [output_list[2], output_list[3][..., 1:2]],
    }

    return formatted_dict


def mesmer_postprocess(model_output, compartment='whole-cell',
                       whole_cell_kwargs=None, nuclear_kwargs=None):
    """Postprocess Mesmer output to generate predictions for distinct cellular compartments

    Args:
        model_output (dict): Output from the Mesmer model. A dict with a key corresponding to
            each cellular compartment with a model prediction. Each key maps to a subsequent dict
            with the following keys entries
            - inner-distance: Prediction for the inner distance transform.
            - outer-distance: Prediction for the outer distance transform
            - fgbg-fg: prediction for the foreground/background transform
            - pixelwise-interior: Prediction for the interior/border/background transform.
        compartment: which cellular compartments to generate predictions for.
            must be one of 'whole_cell', 'nuclear', 'both'
        whole_cell_kwargs (dict): Optional list of post-processing kwargs for whole-cell prediction
        nuclear_kwargs (dict): Optional list of post-processing kwargs for nuclear prediction

    Returns:
        numpy.array: Uniquely labeled mask for each compartment

    Raises:
        ValueError: for invalid compartment flag
    """

    valid_compartments = ['whole-cell', 'nuclear', 'both']

    if whole_cell_kwargs is None:
        whole_cell_kwargs = {}

    if nuclear_kwargs is None:
        nuclear_kwargs = {}

    if compartment not in valid_compartments:
        raise ValueError(f'Invalid compartment supplied: {compartment}. '
                         f'Must be one of {valid_compartments}')

    if compartment == 'whole-cell':
        label_images = deep_watershed(model_output,
                                      **whole_cell_kwargs)
    elif compartment == 'nuclear':
        label_images = deep_watershed(model_output,
                                      **nuclear_kwargs)
    elif compartment == 'both':
        label_images_cell = deep_watershed(model_output,
                                           **whole_cell_kwargs)

        label_images_nucleus = deep_watershed(model_output,
                                              **nuclear_kwargs)

        label_images = np.concatenate([
            label_images_cell,
            label_images_nucleus
        ], axis=1)

    return label_images

def resize_input(image, image_mpp, model_mpp, data_format='channels_first'):
    """Checks if there is a difference between image and model resolution
    and resizes if they are different. Otherwise returns the unmodified
    image.

    Args:
        image (numpy.array): Input image to resize.
        image_mpp (float): Microns per pixel for the ``image``.

    Returns:
        numpy.array: Input image resized if necessary to match ``model_mpp``
    """
    scale_factor = image_mpp / model_mpp

    if data_format == 'channels_first':
        new_shape =  (int(image.shape[-2] * scale_factor),
                        int(image.shape[-1] * scale_factor))
    else:
        new_shape =  (int(image.shape[-2] * scale_factor),
                        int(image.shape[-1] * scale_factor))

    if image_mpp not in {None, model_mpp}:
        image = resize(image, new_shape, data_format=data_format)

    return image

def resize_output(image, original_shape, data_format='channels_first'):
    """Checks if there is a difference between image and model resolution
    and resizes if they are different. Otherwise returns the unmodified
    image.

    Args:
        image (numpy.array): Input image to resize.
        image_mpp (float): Microns per pixel for the ``image``.

    Returns:
        numpy.array: Input image resized if necessary to match ``model_mpp``
    """

    if data_format == 'channels_last':
        B, H, W, C = image.shape
        Bo, Ho, Wo, Co = original_shape
    elif data_format == 'channels_first':
        B, C, H, W = image.shape
        Bo, Co, Ho, Wo = original_shape

    if (H != Ho) | (W != Wo):
        image = resize(image, (Ho, Wo), data_format=data_format)

    return image

def tile_input(image, model_image_shape, pad_mode='constant'):
    """
    Tile the input image to match shape expected by model.
    Expects channels-first format: (B, C, H, W).

    Args:
        image (numpy.array): Input image to tile, must be rank 4 (B, C, H, W).
        model_image_shape (tuple): The (H, W) spatial input size of the model.
        pad_mode (str): The padding mode, one of "constant" or "reflect".

    Returns:
        (numpy.array, dict): Tuple of tiled image and dict of tiling information.

    Raises:
        ValueError: Input images must have only 4 dimensions.
    """
    if image.ndim != 4:
        raise ValueError(
            'tile_image only supports 4D images. '
            f'Image submitted has {image.ndim} dimensions.'
        )

    # Channels-first: spatial dims are axes 2 and 3
    x_diff = image.shape[2] - model_image_shape[0]
    y_diff = image.shape[3] - model_image_shape[1]

    if x_diff < 0 or y_diff < 0:
        # Pad spatial dims only, leave batch and channel dims untouched
        x_diff, y_diff = abs(x_diff), abs(y_diff)
        x_pad = (x_diff // 2, x_diff // 2 + x_diff % 2)
        y_pad = (y_diff // 2, y_diff // 2 + y_diff % 2)

        tiles = np.pad(image, [(0, 0), (0, 0), x_pad, y_pad], 'reflect')
        tiles_info = {'padding': True, 'x_pad': x_pad, 'y_pad': y_pad}
    else:
        tiles, tiles_info = tile_image(
            image,
            model_input_shape=model_image_shape,
            stride_ratio=0.75,
            pad_mode=pad_mode
        )

    return tiles, tiles_info

def untile_output(output_tiles, tiles_info):
    """Untiles either a single array or a list of arrays.
    Expects channels-first format: (B, C, H, W).

    Args:
        output_tiles (numpy.array or list): Array or list of arrays.
        tiles_info (dict): Tiling specs output by the tiling function.
        model_image_shape (tuple): The (H, W) spatial input size of the model.

    Returns:
        numpy.array or list: Untiled image(s) in channels-first format.
    """
    if tiles_info.get('padding', False):
        def _process(im, tiles_info):
            (xl, xh), (yl, yh) = tiles_info['x_pad'], tiles_info['y_pad']
            # Channels-first: spatial dims are axes 2 and 3
            xh = -xh if xh != 0 else None
            yh = -yh if yh != 0 else None
            return im[:, :, xl:xh, yl:yh]
    else:
        def _process(im, tiles_info):
            return untile_image(im, tiles_info)

    if isinstance(output_tiles, list):
        return [_process(o, tiles_info) for o in output_tiles]
    return _process(output_tiles, tiles_info)

def deep_watershed(transforms,
                   radius=10,
                   maxima_threshold=0.1,
                   interior_threshold=0.01,
                   maxima_smooth=0,
                   interior_smooth=1,
                   maxima_index=0,
                   interior_index=-1,
                   label_erosion=0,
                   small_objects_threshold=0,
                   fill_holes_threshold=0,
                   pixel_expansion=None,
                   maxima_algorithm='h_maxima',
                   **kwargs):
    """Uses ``maximas`` and ``interiors`` to perform watershed segmentation.
    ``maximas`` are used as the watershed seeds for each object and
    ``interiors`` are used as the watershed mask.

    Args:
        outputs (list): List of [maximas, interiors] model outputs.
            Use `maxima_index` and `interior_index` if list is longer than 2,
            or if the outputs are in a different order.
        radius (int): Radius of disk used to search for maxima
        maxima_threshold (float): Threshold for the maxima prediction.
        interior_threshold (float): Threshold for the interior prediction.
        maxima_smooth (int): smoothing factor to apply to ``maximas``.
            Use ``0`` for no smoothing.
        interior_smooth (int): smoothing factor to apply to ``interiors``.
            Use ``0`` for no smoothing.
        maxima_index (int): The index of the maxima prediction in ``outputs``.
        interior_index (int): The index of the interior prediction in
            ``outputs``.
        label_erosion (int): Number of pixels to erode segmentation labels.
        small_objects_threshold (int): Removes objects smaller than this size.
        fill_holes_threshold (int): Maximum size for holes within segmented
            objects to be filled.
        pixel_expansion (int): Number of pixels to expand ``interiors``.
        maxima_algorithm (str): Algorithm used to locate peaks in ``maximas``.
            One of ``h_maxima`` (default) or ``peak_local_max``.
            ``peak_local_max`` is much faster but seems to underperform when
            given regious of ambiguous maxima.
    
    Returns:
        numpy.array: Integer label mask for instance segmentation.

    Raises:
        ValueError: ``outputs`` is not properly formatted.
    """

    B, C, H, W = transforms.shape
    
    label_images = np.zeros((B, 1, H, W))
    for batch in range(B):

        maxima = nd.gaussian_filter(transforms[batch, maxima_index], maxima_smooth)
        interior = nd.gaussian_filter(transforms[batch, interior_index], interior_smooth)

        if pixel_expansion:
            interior = dilation(interior, footprint=square(pixel_expansion * 2 + 1))

        # peak_local_max is much faster but has poorer performance
        # when dealing with more ambiguous local maxima
        if maxima_algorithm == 'peak_local_max':
            coords = peak_local_max(
                maxima,
                min_distance=radius,
                threshold_abs=maxima_threshold,
                exclude_border=kwargs.get('exclude_border', False))

            markers = np.zeros_like(maxima)
            slc = tuple(coords[:, i] for i in range(coords.shape[1]))
            markers[slc] = 1
        else:
            # Find peaks and merge equal regions
            markers = h_maxima(image=maxima,
                               h=maxima_threshold,
                               footprint=disk(radius))

        markers = label(markers)
        label_image = watershed(-1 * interior, markers,
                                mask=interior > interior_threshold,
                                watershed_line=0)

        if label_erosion:
            label_image = erode_edges(label_image, label_erosion)

        # Remove small objects
        if small_objects_threshold:
            label_image = remove_small_objects(label_image,
                                               max_size=small_objects_threshold)

        # fill in holes that lie completely within a segmentation label
        if fill_holes_threshold > 0:
            label_image = fill_holes(label_image, size=fill_holes_threshold)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        label_images[batch] = label_image

    return label_images

if __name__ == '__main__':

    default_kwargs_cell = {
        'maxima_threshold': 0.075,
        'maxima_smooth': 0,
        'interior_threshold': 0.2,
        'interior_smooth': 2,
        'small_objects_threshold': 15,
        'fill_holes_threshold': 15,
        'radius': 2,
        'maxima_index': 4,
        'interior_index': 6
    }

    default_kwargs_nuc = {
        'maxima_threshold': 0.1,
        'maxima_smooth': 0,
        'interior_threshold': 0.2,
        'interior_smooth': 2,
        'small_objects_threshold': 15,
        'fill_holes_threshold': 15,
        'radius': 2,
        'maxima_index': 0,
        'interior_index': 2
    }

    ## Test deep watershed for changes to array vs list
    image_mpp = 0.3
    original_shape = (4, 2, 550, 550)
    test_transforms = np.random.random(original_shape)

    resized_image = resize_input(test_transforms, image_mpp, 0.5, data_format='channels_first')
    preprocessed = mesmer_preprocess(resized_image, data_format ='channels_first')
    tiles, tile_info = tile_input(preprocessed, model_image_shape=(256, 256))

    tiles = np.repeat(tiles, 4, axis=1)

    output = untile_output(tiles, tile_info)
    label_image = mesmer_postprocess(output, compartment='both', whole_cell_kwargs=default_kwargs_cell, nuclear_kwargs=default_kwargs_nuc)

    label_image = resize_output(label_image, original_shape, data_format='channels_first')

