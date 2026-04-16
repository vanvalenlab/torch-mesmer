"""Utility functions that may be used in other transforms."""
import logging
import numpy as np

import cv2
import scipy.ndimage as nd

import skimage
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

eps = 1e-7

def resize(data: np.typing.ArrayLike, shape: tuple, data_format='channels_last', labeled_image=False):
    """Resize the data to the given shape.
    Uses openCV to resize the data if the data is a single channel, as it
    is very fast. However, openCV does not support multi-channel resizing,
    so if the data has multiple channels, use skimage.

    Args:
        data (np.array): data to be reshaped. Must have a channel dimension
        shape (tuple): shape of the output data in the form (x,y).
            Batch and channel dimensions are handled automatically and preserved.
        data_format (str): determines the order of the channel axis,
            one of 'channels_first' and 'channels_last'.
        labeled_image (bool): flag to determine how interpolation and floats are handled based
         on whether the data represents raw images or annotations

    Raises:
        ValueError: ndim of data not 3 or 4
        ValueError: Shape for resize can only have length of 2, e.g. (x,y)

    Returns:
        numpy.array: data reshaped to new shape.
    """
    if len(data.shape) not in {3, 4}:
        raise ValueError('Data must have 3 or 4 dimensions, e.g. '
                         '[batch, x, y], [x, y, channel] or '
                         '[batch, x, y, channel]. Input data only has {} '
                         'dimensions.'.format(len(data.shape)))

    if len(shape) != 2:
        raise ValueError('Shape for resize can only have length of 2, e.g. (x,y).'
                         'Input shape has {} dimensions.'.format(len(shape)))

    original_dtype = data.dtype

    # cv2 resize is faster but does not support multi-channel data
    # If the data is multi-channel, use skimage.transform.resize
    channel_axis = 0 if data_format == 'channels_first' else -1
    batch_axis = -1 if data_format == 'channels_first' else 0

    # Use skimage for multichannel data
    if data.shape[channel_axis] > 1:
        # Adjust output shape to account for channel axis
        if data_format == 'channels_first':
            shape = tuple([data.shape[channel_axis]] + list(shape))
        else:
            shape = tuple(list(shape) + [data.shape[channel_axis]])

        # linear interpolation (order 1) for image data, nearest neighbor (order 0) for labels
        # anti_aliasing introduces spurious labels, include only for image data
        order = 0 if labeled_image else 1
        anti_aliasing = not labeled_image

        _resize = lambda d: skimage.transform.resize(d, shape, mode='constant', preserve_range=True,
                                             order=order, anti_aliasing=anti_aliasing)
    # single channel image, resize with cv2
    else:
        shape = tuple(shape)[::-1]  # cv2 expects swapped axes.

        # linear interpolation for image data, nearest neighbor for labels
        # CV2 doesn't support ints for linear interpolation, set to float for image data
        if labeled_image:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR
            data = data.astype('float32')

        _resize = lambda d: np.expand_dims(cv2.resize(np.squeeze(d), shape,
                                                      interpolation=interpolation),
                                           axis=channel_axis)

    # Check for batch dimension to loop over
    if len(data.shape) == 4:
        batch = []
        for i in range(data.shape[batch_axis]):
            d = data[i] if batch_axis == 0 else data[..., i]
            batch.append(_resize(d))
        resized = np.stack(batch, axis=batch_axis)
    else:
        resized = _resize(data)

    return resized.astype(original_dtype)


def fill_holes(label_img: np.typing.ArrayLike, size=10, connectivity=1):
    """Fills holes located completely within a given label with pixels of the same value

    Args:
        label_img (numpy.array): a 2D labeled image
        size (int): maximum size for a hole to be filled in
        connectivity (int): the connectivity used to define the hole

    Returns:
        numpy.array: a labeled image with no holes smaller than ``size``
            contained within any label.
    """
    output_image = np.copy(label_img)

    props = skimage.measure.regionprops(np.squeeze(label_img.astype('int')), cache=False)
    for prop in props:
        if prop.euler_number < 1:

            patch = output_image[prop.slice]

            filled = skimage.morphology.remove_small_holes(
                ar=(patch == prop.label),
                max_size=size,
                connectivity=connectivity)

            output_image[prop.slice] = np.where(filled, prop.label, patch)

    return output_image

def erode_edges(mask: np.typing.ArrayLike, erosion_width):
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


def histogram_normalization(image: np.typing.ArrayLike, kernel_size=None, data_format = 'channels_last'):
    """Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).

    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.

    Args:
        image (numpy.array): numpy array of phase image data.
        kernel_size (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.

    Returns:
        numpy.array: Pre-processed image data with dtype float32.
    """
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    if data_format == 'channels_first':
        image = np.moveaxis(image, 1, -1)

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            X = image[batch, ..., channel]
            sample_value = X[(0,) * X.ndim]
            if (X == sample_value).all():
                # TODO: Deal with constant value arrays
                # https://github.com/scikit-image/scikit-image/issues/4596
                logging.warning('Found constant value array in batch %s and '
                                'channel %s. Normalizing as zeros.',
                                batch, channel)
                image[batch, ..., channel] = np.zeros_like(X)
                continue

            # X = rescale_intensity(X, out_range='float')
            X = skimage.exposure.rescale_intensity(X, out_range=(0.0, 1.0))
            X = skimage.exposure.equalize_adapthist(X, kernel_size=kernel_size)
            image[batch, ..., channel] = X
            
    if data_format == 'channels_first':
        image = np.moveaxis(image, -1, 1)

    return image


def percentile_threshold(image: np.typing.ArrayLike, percentile=99.9):
    """Threshold an image to reduce bright spots

    Args:
        image: numpy array of image data
        percentile: cutoff used to threshold image

    Returns:
        np.array: thresholded version of input image
    """

    processed_image = np.zeros_like(image)
    for img in range(image.shape[0]):
        for chan in range(image.shape[1]):
            current_img = np.copy(image[img, chan])
            non_zero_vals = current_img[np.nonzero(current_img)]

            # only threshold if channel isn't blank
            if len(non_zero_vals) > 0:
                img_max = np.percentile(non_zero_vals, percentile)

                # threshold values down to max
                threshold_mask = current_img > img_max
                current_img[threshold_mask] = img_max

                # update image
                processed_image[img, chan] = current_img

    return processed_image

def pixelwise(prediction: np.typing.ArrayLike, threshold=.8, min_size=50, interior_axis=-2):
    """Post-processing for pixelwise transform predictions.
    Uses the interior predictions to uniquely label every instance.

    Args:
        prediction (numpy.array): pixelwise transform prediction
        threshold (float): confidence threshold for interior predictions
        min_size (int): removes small objects if smaller than min_size.

    Returns:
        numpy.array: post-processed data with each cell uniquely annotated
    """
    # instantiate array to be returned
    labeled_prediction = np.zeros(prediction.shape[:-1] + (1,))

    for batch in range(prediction.shape[0]):
        interior = prediction[[batch], ..., interior_axis] > threshold
        labeled = nd.label(interior)[0]
        labeled = skimage.morphology.remove_small_objects(
            labeled, min_size=min_size, connectivity=1)

        labeled_prediction[batch, ..., 0] = labeled

    return labeled_prediction

def peak_concomp(image: np.typing.ArrayLike, maxima_threshold=0.7):

    '''
    Uses connected components via  ``regionprops`` to find centroids 
    weighted by intensity image.

    Args:
        inputs: (list): list of [maximas, interiors] model outputs.
        maxima_threshold (float): the rough thresholding used to find 
            region masks. Should be /very/ strict.

    Returns:
        markers (numpy.array): Mask of points corresponding to predicted
            centroid 'seeds' that is then passed into the watershedding algorithm.

    '''
    markers = np.zeros_like(image.squeeze())
    maxima_thresh = image > maxima_threshold
    maxima_thresh = skimage.measure.label(maxima_thresh.astype('uint8').squeeze())
    maxima_props = skimage.measure.regionprops(maxima_thresh, intensity_image=image.squeeze())


    for prop in maxima_props:
        x, y = (
            np.around(prop['centroid_weighted'][0]).astype(int), 
            np.around(prop['centroid_weighted'][1]).astype(int)
            )
        
        markers[x,y] = prop['label']

    return markers

def deep_watershed(maximas,
                   interiors,
                   markers=None,
                   radius=10,
                   maxima_threshold=0.1,
                   interior_threshold=0.01,
                   maxima_smooth=0,
                   interior_smooth=1,
                   label_erosion=0,
                   small_objects_threshold=0,
                   fill_holes_threshold=0,
                   pixel_expansion=1,
                   maxima_algorithm='h_maxima',
                   **kwargs):
    """Uses ``maximas`` and ``interiors`` to perform watershed segmentation.
    ``maximas`` are used as the watershed seeds for each object and
    ``interiors`` are used as the watershed mask.

    Args:
        maximas (np.Array): inner distance transform from the model.
        interiors (np.Array): predicted interiors of each object from the model
        markers (np.Array): Optional, found centroids from flow-following. If None, 
            uses method stored in `maxima_algorithm`
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

    # squeeze out the channel dimension if passed
    maxima = nd.gaussian_filter(maximas, maxima_smooth)
    interior = nd.gaussian_filter(interiors, interior_smooth)

    if pixel_expansion:
        fn = skimage.morphology.footprint_rectangle
        interior = skimage.morphology.dilation(interior, footprint=fn((pixel_expansion * 2 + 1, pixel_expansion * 2 + 1)))

    # peak_local_max is much faster but has poorer performance
    # when dealing with more ambiguous local maxima
    if markers is None:
        if maxima_algorithm == 'peak_local_max':
            coords = skimage.feature.peak_local_max(
                maxima,
                min_distance=radius,
                threshold_abs=maxima_threshold,
                exclude_border=kwargs.get('exclude_border', False))

            markers = np.zeros_like(maxima)
            slc = tuple(coords[:, i] for i in range(coords.shape[1]))
            markers[slc] = 1

        elif maxima_algorithm == 'h_maxima':
            # Find peaks and merge equal regions
            fn = skimage.morphology.disk
            markers = skimage.morphology.h_maxima(maxima, h=maxima_threshold, footprint=fn(radius))

        elif maxima_algorithm == 'concomp':           
            # Find peaks and merge equal regions
            markers = peak_concomp(maxima, maxima_threshold=maxima_threshold)

        markers = skimage.measure.label(markers)

    label_image = skimage.segmentation.watershed(-1 * interior, markers,
                            mask=interior > interior_threshold,
                            watershed_line=0)

    if label_erosion:
        label_image = erode_edges(label_image, label_erosion)

    # Remove small objects
    if small_objects_threshold:
        label_image = skimage.morphology.remove_small_objects(label_image,
                                            max_size=small_objects_threshold)

    # fill in holes that lie completely within a segmentation label
    if fill_holes_threshold > 0:
        label_image = fill_holes(label_image, size=fill_holes_threshold)

    # Relabel the label image
    label_image, _, _ = skimage.segmentation.relabel_sequential(label_image)


    return label_image, markers

def create_sample_overlay(labels, transforms):

    transform_names = ['inner', 'pixel1','pixel2','bg']

    # 4, H, W

    labels = labels.cpu().numpy()
    transforms = transforms.cpu().numpy()

    fig, ax = plt.subplots(2, labels.shape[0])

    for i in range(transforms.shape[0]):
        ax[0, i].imshow(transforms[i], vmin=0, vmax=1)
        ax[0,i].set_title(f"Predicted {transform_names[i]}")
        ax[0, i].set_axis_off()
    
    for i in range(labels.shape[0]):
        ax[1, i].imshow(labels[i], vmin=0, vmax=1)
        ax[1,i].set_title(f"True {transform_names[i]}")
        ax[1, i].set_axis_off()

    fig.tight_layout()

    return fig
   
def percentile_threshold(image, percentile=99.9):
    """Threshold an image to reduce bright spots

    Args:
        image: numpy array of image data
        percentile: cutoff used to threshold image

    Returns:
        np.array: thresholded version of input image
    """

    processed_image = np.zeros_like(image)
    for img in range(image.shape[0]):
        for chan in range(image.shape[1]):
            current_img = np.copy(image[img, chan])
            non_zero_vals = current_img[np.nonzero(current_img)]

            # only threshold if channel isn't blank
            if len(non_zero_vals) > 0:
                img_max = np.percentile(non_zero_vals, percentile)

                # threshold values down to max
                threshold_mask = current_img > img_max
                current_img[threshold_mask] = img_max

                # update image
                processed_image[img, chan] = current_img

    return processed_image

def compute_overlap_vectorized(boxes, query_boxes):
    """
    Vectorized computation of IoU overlaps.
    
    Args
        boxes: (N, 4) ndarray of float - format [x1, y1, x2, y2]
        query_boxes: (K, 4) ndarray of float - format [x1, y1, x2, y2]

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    
    # Compute areas
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_area = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    
    # Broadcast to compute intersections
    # boxes: (N, 1, 4), query_boxes: (1, K, 4)
    iw = (np.minimum(boxes[:, None, 2], query_boxes[None, :, 2]) - 
          np.maximum(boxes[:, None, 0], query_boxes[None, :, 0]) + 1)
    ih = (np.minimum(boxes[:, None, 3], query_boxes[None, :, 3]) - 
          np.maximum(boxes[:, None, 1], query_boxes[None, :, 1]) + 1)
    
    # Clip to 0
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    
    # Compute intersection and union
    intersection = iw * ih
    union = boxes_area[:, None] + query_area[None, :] - intersection
    
    # Compute IoU
    overlaps = intersection / union
    
    return overlaps

def cast_to_tuple(x):
    try:
        tup_x = tuple(x)
    except TypeError:
        tup_x = () if x is None else (x,)
    return tup_x

def get_box_labels(arr):
    """Get the bounding box and label for all objects in the image.

    Args:
        arr (np.array): integer label array of objects.

    Returns:
        tuple(list(np.array), list(int)): A tuple of bounding boxes and
            the corresponding integer labels.
    """
    props = skimage.measure.regionprops(np.squeeze(arr.astype('int')), cache=False)
    boxes, labels = [], []
    for prop in props:
        boxes.append(np.array(prop.bbox))
        labels.append(int(prop.label))
    boxes = np.array(boxes).astype('double')
    return boxes, labels

def match_nodes(y_true, y_pred):
    """Loads all data that matches each pattern and compares the graphs.

    Args:
        y_true (numpy.array): ground truth array with all cells labeled uniquely.
        y_pred (numpy.array): data array to match to unique.

    Returns:
        numpy.array: IoU of ground truth cells and predicted cells.
    """
    num_frames = y_true.shape[0]
    # TODO: does max make the shape bigger than necessary?
    iou = np.zeros((num_frames, np.max(y_true) + 1, np.max(y_pred) + 1))

    # Compute IOUs only when neccesary
    # If bboxs for true and pred do not overlap with each other, the assignment
    # is immediate. Otherwise use pixelwise IOU to determine which cell is which

    # Regionprops expects one frame at a time
    for frame in range(num_frames):
        gt_frame = y_true[frame]
        res_frame = y_pred[frame]

        gt_props = skimage.measure.regionprops(np.squeeze(gt_frame.astype('int')))
        gt_boxes = [np.array(gt_prop.bbox) for gt_prop in gt_props]
        gt_boxes = np.array(gt_boxes).astype('double')
        gt_box_labels = [int(gt_prop.label) for gt_prop in gt_props]

        res_props = skimage.measure.regionprops(np.squeeze(res_frame.astype('int')))
        res_boxes = [np.array(res_prop.bbox) for res_prop in res_props]
        res_boxes = np.array(res_boxes).astype('double')
        res_box_labels = [int(res_prop.label) for res_prop in res_props]

        # has the form [gt_bbox, res_bbox]
        overlaps = compute_overlap_vectorized(gt_boxes, res_boxes)

        # Find the bboxes that have overlap at all
        # (ind_ corresponds to box number - starting at 0)
        ind_gt, ind_res = np.nonzero(overlaps)

        # frame_ious = np.zeros(overlaps.shape)
        for index in range(ind_gt.shape[0]):
            iou_gt_idx = gt_box_labels[ind_gt[index]]
            iou_res_idx = res_box_labels[ind_res[index]]
            intersection = np.logical_and(
                gt_frame == iou_gt_idx, res_frame == iou_res_idx)
            union = np.logical_or(
                gt_frame == iou_gt_idx, res_frame == iou_res_idx)
            iou[frame, iou_gt_idx, iou_res_idx] = intersection.sum() / union.sum()

    return iou

def split_stack(arr, batch, n_split1, axis1, n_split2, axis2):
    """Crops an array in the width and height dimensions to produce
    a stack of smaller arrays

    Args:
        arr (numpy.array): Array to be split with at least 2 dimensions
        batch (bool): True if the zeroth dimension of arr is a batch or
            frame dimension
        n_split1 (int): Number of sections to produce from the first split axis
            Must be able to divide arr.shape[axis1] evenly by n_split1
        axis1 (int): Axis on which to perform first split
        n_split2 (int): Number of sections to produce from the second split axis
            Must be able to divide arr.shape[axis2] evenly by n_split2
        axis2 (int): Axis on which to perform first split

    Returns:
        numpy.array: Array after dual splitting with frames in the zeroth dimension

    Raises:
        ValueError: arr.shape[axis] must be evenly divisible by n_split
            for both the first and second split

    Examples:
        >>> from deepcell import metrics
        >>> from numpy import np
        >>> arr = np.ones((10, 100, 100, 1))
        >>> out = metrics.split_stack(arr, True, 10, 1, 10, 2)
        >>> out.shape
        (1000, 10, 10, 1)
        >>> arr = np.ones((100, 100, 1))
        >>> out = metrics.split_stack(arr, False, 10, 1, 10, 2)
        >>> out.shape
        (100, 10, 10, 1)
    """
    # Check that n_split will divide equally
    if ((arr.shape[axis1] % n_split1) != 0) | ((arr.shape[axis2] % n_split2) != 0):
        raise ValueError(
            'arr.shape[axis] must be evenly divisible by n_split'
            'for both the first and second split')

    split1 = np.split(arr, n_split1, axis=axis1)

    # If batch dimension doesn't exist, create and adjust axis2
    if batch is False:
        split1con = np.stack(split1)
        axis2 += 1
    else:
        split1con = np.concatenate(split1, axis=0)

    split2 = np.split(split1con, n_split2, axis=axis2)
    split2con = np.concatenate(split2, axis=0)

    return split2con

def merge_nearby_points(points, r):
    """
    Merge points within distance r by averaging coordinates.
    
    Parameters:
    -----------
    points : array-like, shape (n, 2)
        Array of (x, y) coordinates
    distance_matrix : array-like, shape (n, n)
        Pairwise distance matrix
    r : float
        Distance threshold for merging
    
    Returns:
    --------
    merged_points : ndarray, shape (m, 2)
        Array of merged point coordinates
    labels : ndarray, shape (n,)
        Cluster label for each original point
    """
    points = np.asarray(points)

    distance_matrix = cdist(points, points)    
    # Create adjacency matrix: points are connected if distance <= r
    adjacency = distance_matrix <= r
    
    # Find connected components
    n_components, labels = connected_components(
        csgraph=csr_matrix(adjacency),
        directed=False
    )
    
    # Merge points in each component by averaging
    merged_points = np.zeros((n_components, 2))
    for i in range(n_components):
        mask = labels == i
        merged_points[i] = points[mask].mean(axis=0)
    
    merged_points = merged_points.astype(int)
    
    return merged_points