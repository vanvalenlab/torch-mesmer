import numpy as np
from skimage.measure import regionprops

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

def _cast_to_tuple(x):
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
    props = regionprops(np.squeeze(arr.astype('int')), cache=False)
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

        gt_props = regionprops(np.squeeze(gt_frame.astype('int')))
        gt_boxes = [np.array(gt_prop.bbox) for gt_prop in gt_props]
        gt_boxes = np.array(gt_boxes).astype('double')
        gt_box_labels = [int(gt_prop.label) for gt_prop in gt_props]

        res_props = regionprops(np.squeeze(res_frame.astype('int')))
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