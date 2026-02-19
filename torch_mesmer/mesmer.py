import torch
torch.set_num_threads(24)

import numpy as np

from torch_mesmer.utils import histogram_normalization, percentile_threshold, deep_watershed, resize, erode_edges, fill_holes
from torch_mesmer.model import PanopticNet
import warnings

import scipy.ndimage as nd
from scipy.signal import windows

from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import remove_small_objects, h_maxima
from skimage.morphology import disk, ball, square, cube, dilation
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
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    window_x = spline_window(window_size[0], overlap_x[0], overlap_x[1], power=power)
    window_y = spline_window(window_size[1], overlap_y[0], overlap_y[1], power=power)

    window_x = np.expand_dims(np.expand_dims(window_x, -1), -1)
    window_y = np.expand_dims(np.expand_dims(window_y, -1), -1)

    window = window_x * window_y.transpose(1, 0, 2)
    return window


def deep_watershed(outputs,
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
    try:
        maximas = outputs[maxima_index]
        interiors = outputs[interior_index]
    except (TypeError, KeyError, IndexError):
        raise ValueError('`outputs` should be a list of at least two '
                         'NumPy arryas of equal shape.')

    valid_algos = {'h_maxima', 'peak_local_max'}
    if maxima_algorithm not in valid_algos:
        raise ValueError('Invalid value for maxima_algorithm: {}. '
                         'Must be one of {}'.format(
                             maxima_algorithm, valid_algos))

    total_pixels = maximas.shape[1] * maximas.shape[2]
    if maxima_algorithm == 'h_maxima' and total_pixels > 5000**2:
        warnings.warn('h_maxima peak finding algorithm was selected, '
                      'but the provided image is larger than 5k x 5k pixels.'
                      'This will lead to slow prediction performance.')
    # Handle deprecated arguments
    min_distance = kwargs.pop('min_distance', None)
    if min_distance is not None:
        radius = min_distance
        warnings.warn('`min_distance` is now deprecated in favor of `radius`. '
                      'The value passed for `radius` will be used.',
                      DeprecationWarning)

    # distance_threshold vs interior_threshold
    distance_threshold = kwargs.pop('distance_threshold', None)
    if distance_threshold is not None:
        interior_threshold = distance_threshold
        warnings.warn('`distance_threshold` is now deprecated in favor of '
                      '`interior_threshold`. The value passed for '
                      '`distance_threshold` will be used.',
                      DeprecationWarning)

    # detection_threshold vs maxima_threshold
    detection_threshold = kwargs.pop('detection_threshold', None)
    if detection_threshold is not None:
        maxima_threshold = detection_threshold
        warnings.warn('`detection_threshold` is now deprecated in favor of '
                      '`maxima_threshold`. The value passed for '
                      '`detection_threshold` will be used.',
                      DeprecationWarning)

    if maximas.shape[:-1] != interiors.shape[:-1]:
        raise ValueError('All input arrays must have the same shape. '
                         'Got {} and {}'.format(
                             maximas.shape, interiors.shape))

    if maximas.ndim not in {4, 5}:
        raise ValueError('maxima and interior tensors must be rank 4 or 5. '
                         'Rank 4 is 2D data of shape (batch, x, y, c). '
                         'Rank 5 is 3D data of shape (batch, frames, x, y, c).')

    input_is_3d = maximas.ndim > 4

    # fill_holes is not supported in 3D
    if fill_holes_threshold and input_is_3d:
        warnings.warn('`fill_holes` is not supported for 3D data.')
        fill_holes_threshold = 0

    label_images = []
    for maxima, interior in zip(maximas, interiors):
        # squeeze out the channel dimension if passed
        maxima = nd.gaussian_filter(maxima[..., 0], maxima_smooth)
        interior = nd.gaussian_filter(interior[..., 0], interior_smooth)

        if pixel_expansion:
            fn = cube if input_is_3d else square
            interior = dilation(interior, footprint=fn(pixel_expansion * 2 + 1))

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
            fn = ball if input_is_3d else disk
            markers = h_maxima(image=maxima,
                               h=maxima_threshold,
                               footprint=fn(radius))

        markers = label(markers)
        label_image = watershed(-1 * interior, markers,
                                mask=interior > interior_threshold,
                                watershed_line=0)

        if label_erosion:
            label_image = erode_edges(label_image, label_erosion)

        # Remove small objects
        if small_objects_threshold:
            label_image = remove_small_objects(label_image,
                                               min_size=small_objects_threshold)

        # fill in holes that lie completely within a segmentation label
        if fill_holes_threshold > 0:
            label_image = fill_holes(label_image, size=fill_holes_threshold)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        label_images.append(label_image)

    label_images = np.stack(label_images, axis=0)
    label_images = np.expand_dims(label_images, axis=-1)

    return label_images

def untile_image(tiles, tiles_info, power=2, **kwargs):
    """Untile a set of tiled images back to the original model shape.

     Args:
         tiles (numpy.array): The tiled images image to untile.
         tiles_info (dict): Details of how the image was tiled (from tile_image).
         power (int): The power of the window function

     Returns:
         numpy.array: The untiled image.
     """
    # Define mininally acceptable tile_size and stride_ratio for spline interpolation
    min_tile_size = 32
    min_stride_ratio = 0.5

    stride_ratio = tiles_info['stride_ratio']
    image_shape = tiles_info['image_shape']
    batches = tiles_info['batches']
    x_starts = tiles_info['x_starts']
    x_ends = tiles_info['x_ends']
    y_starts = tiles_info['y_starts']
    y_ends = tiles_info['y_ends']
    overlaps_x = tiles_info['overlaps_x']
    overlaps_y = tiles_info['overlaps_y']
    tile_size_x = tiles_info['tile_size_x']
    tile_size_y = tiles_info['tile_size_y']
    stride_ratio = tiles_info['stride_ratio']
    x_pad = tiles_info['pad_x']
    y_pad = tiles_info['pad_y']

    image_shape = [image_shape[0], image_shape[1], image_shape[2], tiles.shape[-1]]
    window_size = (tile_size_x, tile_size_y)
    image = np.zeros(image_shape, dtype=float)

    window_cache = {}
    for x, y in zip(overlaps_x, overlaps_y):
        if (x, y) not in window_cache:
            w = window_2D(window_size, overlap_x=x, overlap_y=y, power=power)
            window_cache[(x, y)] = w

    for tile, batch, x_start, x_end, y_start, y_end, overlap_x, overlap_y in zip(
            tiles, batches, x_starts, x_ends, y_starts, y_ends, overlaps_x, overlaps_y):

        # Conditions under which to use spline interpolation
        # A tile size or stride ratio that is too small gives inconsistent results,
        # so in these cases we skip interpolation and just return the raw tiles
        if (min_tile_size <= tile_size_x < image_shape[1] and
                min_tile_size <= tile_size_y < image_shape[2] and
                stride_ratio >= min_stride_ratio):
            window = window_cache[(overlap_x, overlap_y)]
            image[batch, x_start:x_end, y_start:y_end, :] += tile * window
        else:
            image[batch, x_start:x_end, y_start:y_end, :] = tile

    image = image.astype(tiles.dtype)

    x_start = x_pad[0]
    y_start = y_pad[0]
    x_end = image_shape[1] - x_pad[1]
    y_end = image_shape[2] - y_pad[1]

    image = image[:, x_start:x_end, y_start:y_end, :]

    return image

def tile_image(image, model_input_shape=(512, 512),
               stride_ratio=0.75, pad_mode='constant'):
    """
    Tile large image into many overlapping tiles of size "model_input_shape".

    Args:
        image (numpy.array): The image to tile, must be rank 4.
        model_input_shape (tuple): The input size of the model.
        stride_ratio (float): The stride expressed as a fraction of the tile size.
        pad_mode (str): Padding mode passed to ``np.pad``.

    Returns:
        tuple: (numpy.array, dict): A tuple consisting of an array of tiled
            images and a dictionary of tiling details (for use in un-tiling).

    Raises:
        ValueError: image is not rank 4.
    """
    if image.ndim != 4:
        raise ValueError('Expected image of rank 4, got {}'.format(image.ndim))

    image_size_x, image_size_y = image.shape[1:3]
    tile_size_x = model_input_shape[0]
    tile_size_y = model_input_shape[1]

    ceil = lambda x: int(np.ceil(x))
    round_to_even = lambda x: int(np.ceil(x / 2.0) * 2)

    stride_x = min(round_to_even(stride_ratio * tile_size_x), tile_size_x)
    stride_y = min(round_to_even(stride_ratio * tile_size_y), tile_size_y)

    rep_number_x = max(ceil((image_size_x - tile_size_x) / stride_x + 1), 1)
    rep_number_y = max(ceil((image_size_y - tile_size_y) / stride_y + 1), 1)
    new_batch_size = image.shape[0] * rep_number_x * rep_number_y

    tiles_shape = (new_batch_size, tile_size_x, tile_size_y, image.shape[3])
    tiles = np.zeros(tiles_shape, dtype=image.dtype)

    # Calculate overlap of last tile
    overlap_x = (tile_size_x + stride_x * (rep_number_x - 1)) - image_size_x
    overlap_y = (tile_size_y + stride_y * (rep_number_y - 1)) - image_size_y

    # Calculate padding needed to account for overlap and pad image accordingly
    pad_x = (int(np.ceil(overlap_x / 2)), int(np.floor(overlap_x / 2)))
    pad_y = (int(np.ceil(overlap_y / 2)), int(np.floor(overlap_y / 2)))
    pad_null = (0, 0)
    padding = (pad_null, pad_x, pad_y, pad_null)
    image = np.pad(image, padding, pad_mode)

    counter = 0
    batches = []
    x_starts = []
    x_ends = []
    y_starts = []
    y_ends = []
    overlaps_x = []
    overlaps_y = []

    for b in range(image.shape[0]):
        for i in range(rep_number_x):
            for j in range(rep_number_y):
                x_axis = 1
                y_axis = 2

                # Compute the start and end for each tile
                if i != rep_number_x - 1:  # not the last one
                    x_start, x_end = i * stride_x, i * stride_x + tile_size_x
                else:
                    x_start, x_end = image.shape[x_axis] - tile_size_x, image.shape[x_axis]

                if j != rep_number_y - 1:  # not the last one
                    y_start, y_end = j * stride_y, j * stride_y + tile_size_y
                else:
                    y_start, y_end = image.shape[y_axis] - tile_size_y, image.shape[y_axis]

                # Compute the overlaps for each tile
                if i == 0:
                    overlap_x = (0, tile_size_x - stride_x)
                elif i == rep_number_x - 2:
                    overlap_x = (tile_size_x - stride_x, tile_size_x - image.shape[x_axis] + x_end)
                elif i == rep_number_x - 1:
                    overlap_x = ((i - 1) * stride_x + tile_size_x - x_start, 0)
                else:
                    overlap_x = (tile_size_x - stride_x, tile_size_x - stride_x)

                if j == 0:
                    overlap_y = (0, tile_size_y - stride_y)
                elif j == rep_number_y - 2:
                    overlap_y = (tile_size_y - stride_y, tile_size_y - image.shape[y_axis] + y_end)
                elif j == rep_number_y - 1:
                    overlap_y = ((j - 1) * stride_y + tile_size_y - y_start, 0)
                else:
                    overlap_y = (tile_size_y - stride_y, tile_size_y - stride_y)

                tiles[counter] = image[b, x_start:x_end, y_start:y_end, :]
                batches.append(b)
                x_starts.append(x_start)
                x_ends.append(x_end)
                y_starts.append(y_start)
                y_ends.append(y_end)
                overlaps_x.append(overlap_x)
                overlaps_y.append(overlap_y)
                counter += 1

    tiles_info = {}
    tiles_info['batches'] = batches
    tiles_info['x_starts'] = x_starts
    tiles_info['x_ends'] = x_ends
    tiles_info['y_starts'] = y_starts
    tiles_info['y_ends'] = y_ends
    tiles_info['overlaps_x'] = overlaps_x
    tiles_info['overlaps_y'] = overlaps_y
    tiles_info['stride_x'] = stride_x
    tiles_info['stride_y'] = stride_y
    tiles_info['tile_size_x'] = tile_size_x
    tiles_info['tile_size_y'] = tile_size_y
    tiles_info['stride_ratio'] = stride_ratio
    tiles_info['image_shape'] = image.shape
    tiles_info['dtype'] = image.dtype
    tiles_info['pad_x'] = pad_x
    tiles_info['pad_y'] = pad_y

    return tiles, tiles_info


# pre- and post-processing functions
def mesmer_preprocess(image, **kwargs):
    """Preprocess input data for Mesmer model.

    Args:
        image: array to be processed

    Returns:
        np.array: processed image array
    """

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
        label_images = deep_watershed(model_output['whole-cell'],
                                      **whole_cell_kwargs)
    elif compartment == 'nuclear':
        label_images = deep_watershed(model_output['nuclear'],
                                      **nuclear_kwargs)
    elif compartment == 'both':
        label_images_cell = deep_watershed(model_output['whole-cell'],
                                           **whole_cell_kwargs)

        label_images_nucleus = deep_watershed(model_output['nuclear'],
                                              **nuclear_kwargs)

        label_images = np.concatenate([
            label_images_cell,
            label_images_nucleus
        ], axis=-1)

    else:
        raise ValueError(f'Invalid compartment supplied: {compartment}. '
                         f'Must be one of {valid_compartments}')

    return label_images

def resize_input(image, image_mpp, model_mpp):
    """Checks if there is a difference between image and model resolution
    and resizes if they are different. Otherwise returns the unmodified
    image.

    Args:
        image (numpy.array): Input image to resize.
        image_mpp (float): Microns per pixel for the ``image``.

    Returns:
        numpy.array: Input image resized if necessary to match ``model_mpp``
    """
    # Don't scale the image if mpp is the same or not defined
    if image_mpp not in {None, model_mpp}:
        shape = image.shape
        scale_factor = image_mpp / model_mpp
        new_shape = (int(shape[1] * scale_factor),
                        int(shape[2] * scale_factor))
        image = resize(image, new_shape, data_format='channels_last')
    return image

def resize_output(image, original_shape):
        """Rescales input if the shape does not match the original shape
        excluding the batch and channel dimensions.

        Args:
            image (numpy.array): Image to be rescaled to original shape
            original_shape (tuple): Shape of the original input image

        Returns:
            numpy.array: Rescaled image
        """
        if not isinstance(image, list):
            image = [image]

        for i in range(len(image)):
            img = image[i]
            # Compare x,y based on rank of image
            # Check if unnecessary
            if len(img.shape) == 4:
                same = img.shape[1:-1] == original_shape[1:-1]
            elif len(img.shape) == 3:
                same = img.shape[1:] == original_shape[1:-1]
            else:
                same = img.shape == original_shape[1:-1]

            # Resize if same is false
            if not same:
                # Resize function only takes the x,y dimensions for shape
                new_shape = original_shape[1:-1]
                img = resize(img, new_shape,
                             data_format='channels_last',
                             labeled_image=True)
            image[i] = img

        if len(image) == 1:
            image = image[0]

        return image

def tile_input(image, model_image_shape, pad_mode='constant'):
    """Tile the input image to match shape expected by model
    using the ``deepcell_toolbox`` or ``toolbox_utils`` function.

    Only supports 4D images.

    Args:
        image (numpy.array): Input image to tile
        pad_mode (str): The padding mode, one of "constant" or "reflect".

    Raises:
        ValueError: Input images must have only 4 dimensions

    Returns:
        (numpy.array, dict): Tuple of tiled image and dict of tiling
        information.
    """
    if len(image.shape) != 4:
        raise ValueError('toolbox_utils.tile_image only supports 4d images.'
                            f'Image submitted for predict has {len(image.shape)} dimensions')

    # Check difference between input and model image size
    x_diff = image.shape[1] - model_image_shape[0]
    y_diff = image.shape[2] - model_image_shape[1]

    # Check if the input is smaller than model image size
    if x_diff < 0 or y_diff < 0:
        # Calculate padding
        x_diff, y_diff = abs(x_diff), abs(y_diff)
        x_pad = (x_diff // 2, x_diff // 2 + 1) if x_diff % 2 else (x_diff // 2, x_diff // 2)
        y_pad = (y_diff // 2, y_diff // 2 + 1) if y_diff % 2 else (y_diff // 2, y_diff // 2)

        tiles = np.pad(image, [(0, 0), x_pad, y_pad, (0, 0)], 'reflect')
        tiles_info = {'padding': True,
                        'x_pad': x_pad,
                        'y_pad': y_pad}
    # Otherwise tile images larger than model size
    else:
        # Tile images, needs 4d
        tiles, tiles_info = tile_image(image, model_input_shape=model_image_shape,
                                        stride_ratio=0.75, pad_mode=pad_mode)

    return tiles, tiles_info

def batch_predict(tiles, batch_size, model, device):
    """Batch process tiles to generate model predictions.

    Batch processing occurs without loading entire image stack onto
    GPU memory, a problem that exists in other solutions such as
    keras.predict.

    Args:
        tiles (numpy.array): Tiled data which will be fed to model
        batch_size (int): Number of images to predict on per batch

    Returns:
        list: Model outputs
    """

    # list to hold final output
    output_tiles = []

    model.eval()
    batch_outputs_list = []

    for i in range(0, tiles.shape[0], batch_size):
        batch_inputs = tiles[i:i + batch_size, ...]
        temp_input = torch.tensor(batch_inputs).to(device)
        temp_input = torch.permute(temp_input, (0, 3, 1, 2))    

        with torch.inference_mode():
            outs = model(temp_input)

        batch_outputs_list.append([torch.permute(k, (0, 2, 3, 1)) for k in outs])
        
    for i, batch_outputs in enumerate(batch_outputs_list):

        # model with only a single output gets temporarily converted to a list
        if not isinstance(batch_outputs, list):
            batch_outputs = [batch_outputs.cpu().detach()]

        else:
            batch_outputs = [b_out.cpu().detach() for b_out in batch_outputs]

        # initialize output list with empty arrays to hold all batches
        if not output_tiles:
            for batch_out in batch_outputs:
                shape = (tiles.shape[0],) + batch_out.shape[1:]
                output_tiles.append(np.zeros(shape, dtype=tiles.dtype))

        # save each batch to corresponding index in output list
        for j, batch_out in enumerate(batch_outputs):
            output_tiles[j][i*batch_size:(i+1) * batch_size, ...] = batch_out

    return output_tiles

def untile_output(output_tiles, tiles_info, model_image_shape):
    """Untiles either a single array or a list of arrays
    according to a dictionary of tiling specs

    Args:
        output_tiles (numpy.array or list): Array or list of arrays.
        tiles_info (dict): Tiling specs output by the tiling function.

    Returns:
        numpy.array or list: Array or list according to input with untiled images
    """
    # If padding was used, remove padding
    if tiles_info.get('padding', False):
        def _process(im, tiles_info):
            ((xl, xh), (yl, yh)) = tiles_info['x_pad'], tiles_info['y_pad']
            # Edge-case: upper-bound == 0 - this can occur when only one of
            # either X or Y is smaller than model_img_shape while the other
            # is equal to model_image_shape.
            xh = -xh if xh != 0 else None
            yh = -yh if yh != 0 else None
            return im[:, xl:xh, yl:yh, :]
    # Otherwise untile
    else:
        def _process(im, tiles_info):
            out = untile_image(im, tiles_info, model_input_shape=model_image_shape)
            return out

    if isinstance(output_tiles, list):
        output_images = [_process(o, tiles_info) for o in output_tiles]
    else:
        output_images = _process(output_tiles, tiles_info)

    return output_images

class Mesmer():

    def __init__(
            self, 
            model_path=None, 
            device=None, 
            n_semantic_classes=[1,3,1,3]
    ):

        if device is None:
            self.device = 'cpu'
        else:
            self.device=device

        if model_path is None:
            raise Exception("Please provide a path to the model checkpoint file.")
        
        print("Initializing model...")
        self.model = PanopticNet(
            crop_size=256,
            backbone='resnet50',
            pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
            backbone_levels=['C3', 'C4', 'C5'],
            n_semantic_classes=n_semantic_classes
        ).to(self.device).eval()

        # Dummy data to make semantic heads
        dummy = torch.rand(1, 2, self.model.crop_size, self.model.crop_size).to(self.device)
        _ = self.model(dummy)
        del dummy

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        # Whole cell, nuc

        self.model_image_shape = (256,256, 2)
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.required_rank = len(self.model_image_shape) + 1
        self.required_channels = self.model_image_shape[-1]

        self.model_mpp = 0.5
        
        self.default_kwargs = {
            'w':{
                'small_objects_threshold': 15,
                'fill_holes_threshold': 15,
                'n_iter': 200,
                'step_size': 0.1,
                'maxima_threshold': 0.15,
                'maxima_smooth': 0,
                'interior_threshold': 0.2,
                'interior_smooth': 0.5,
                'radius': 2,
                'maxima_algorithm': 'h_maxima',
                'postprocess_method': 'classical',
                'relevant_votes': 2,
                'merge_radius': 10},

            'n': {
                'small_objects_threshold': 15,
                'fill_holes_threshold': 15,
                'n_iter': 200,
                'step_size': 0.1,
                'maxima_threshold': 0.15,
                'maxima_smooth': 0,
                'interior_threshold': 0.3,
                'interior_smooth': 0.5,
                'radius': 2,
                'maxima_algorithm': 'h_maxima',
                'postprocess_method': 'classical',
                'relevant_votes': 2,
                'merge_radius': 5}
            }
        
    def predict(self,
                image,
                batch_size=4,
                image_mpp=None,
                preprocess_kwargs={},
                compartment='whole-cell',
                pad_mode='constant',
                postprocess_kwargs_whole_cell={},
                postprocess_kwargs_nuclear={}):
        """Generates a labeled image of the input running prediction with
        appropriate pre and post processing functions.

        Input images are required to have 4 dimensions
        ``[batch, x, y, channel]``.
        Additional empty dimensions can be added using ``np.expand_dims``.

        Args:
            image (numpy.array): Input image with shape
                ``[batch, x, y, channel]``.
            batch_size (int): Number of images to predict on per batch.
            image_mpp (float): Microns per pixel for ``image``.
            compartment (str): Specify type of segmentation to predict.
                Must be one of ``"whole-cell"``, ``"nuclear"``, ``"both"``.
            preprocess_kwargs (dict): Keyword arguments to pass to the
                pre-processing function.
            postprocess_kwargs (dict): Keyword arguments to pass to the
                post-processing function.

        Raises:
            ValueError: Input data must match required rank of the application,
                calculated as one dimension more (batch dimension) than expected
                by the model.

            ValueError: Input data must match required number of channels.

        Returns:
            numpy.array: Instance segmentation mask.
        """
        default_kwargs_cell = {
            'maxima_threshold': 0.075,
            'maxima_smooth': 0,
            'interior_threshold': 0.2,
            'interior_smooth': 2,
            'small_objects_threshold': 15,
            'fill_holes_threshold': 15,
            'radius': 2
        }

        default_kwargs_nuc = {
            'maxima_threshold': 0.1,
            'maxima_smooth': 0,
            'interior_threshold': 0.2,
            'interior_smooth': 2,
            'small_objects_threshold': 15,
            'fill_holes_threshold': 15,
            'radius': 2
        }

        # overwrite defaults with any user-provided values
        postprocess_kwargs_whole_cell = {**default_kwargs_cell,
                                         **postprocess_kwargs_whole_cell}

        postprocess_kwargs_nuclear = {**default_kwargs_nuc,
                                      **postprocess_kwargs_nuclear}

        # create dict to hold all of the post-processing kwargs
        postprocess_kwargs = {
            'whole_cell_kwargs': postprocess_kwargs_whole_cell,
            'nuclear_kwargs': postprocess_kwargs_nuclear,
            'compartment': compartment
        }

        # Keep track of original shape for rescaling after processing
        orig_img_shape = image.shape

        resized_image = resize_input(image, image_mpp, 0.5)

        image = mesmer_preprocess(resized_image, **preprocess_kwargs)

        # Tile images, raises error if the image is not 4d
        tiles, tiles_info = tile_input(image, pad_mode=pad_mode, model_image_shape=self.model_image_shape)

        output_tiles = []
        tiles = np.moveaxis(tiles, -1, 1)       
        for tile_batch_start in range(0, tiles.shape[0], batch_size):
            # Load only this batch to GPU
            tile_batch = torch.tensor(tiles[tile_batch_start:tile_batch_start+batch_size]).to(self.device)
            
            with torch.inference_mode():
                pred = self.model(tile_batch)
                        
            # Move predictions back to CPU to save GPU memory
            output_tiles.append(pred.cpu())
            
            # Clean up GPU memory
            del tile_batch, pred
            if self.device != 'cpu':
                torch.cuda.empty_cache()

        output_tiles = np.concat(output_tiles, axis=0)
        output_tiles = np.moveaxis(output_tiles, 1,-1)

        # Untile images
        output_images = untile_output(output_tiles, tiles_info, self.model_image_shape)

        formatted_dict = {
            'whole-cell': [output_images[:,..., 4:5], output_images[:,..., 6:7]],
            'nuclear': [output_images[:,..., 0:1], output_images[:,..., 2:3]],
        }

        # output_images = format_output_mesmer(output_images)
        label_image = mesmer_postprocess(formatted_dict, **postprocess_kwargs)
        # Restore channel dimension if not already there
        # TODO: check if unnecessary
        if len(image.shape) == self.required_rank - 1:
            image = np.expand_dims(image, axis=-1)

        label_image = resize_output(label_image, orig_img_shape)
        return label_image, output_images
