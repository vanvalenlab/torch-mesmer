"""Custom metrics for pixel-based and object-based classification accuracy.

The schema for this analysis was adopted from the description of object-based
statistics in Caicedo et al. (2018) Evaluation of Deep Learning Strategies for
Nucleus Segmentation in Fluorescence Images. BioRxiv 335216.

The SEG metric was adapted from Maska et al. (2014). A benchmark for comparison
of cell tracking algorithms. Bioinformatics 30, 1609-1617.

The linear classification schema used to match objects in truth and prediction
frames was adapted from Jaqaman et al. (2008). Robust single-particle tracking
in live-cell time-lapse sequences. Nature Methods 5, 695-702.
"""

import datetime
import json
import logging
import os
import warnings

import numpy as np
import pandas as pd
import networkx as nx

from scipy.optimize import linear_sum_assignment
from scipy.stats import hmean
from skimage.segmentation import relabel_sequential
from tqdm import tqdm
from dataclasses import dataclass

from torch_mesmer.utils import compute_overlap_vectorized, get_box_labels, cast_to_tuple

@dataclass
class Detection():  # pylint: disable=useless-object-inheritance
    """Object to hold relevant information about a given detection."""
    true_index: int
    pred_index: int

    def __init__(self, true_index=None, pred_index=None):
        # cast the indices as tuples if possible to make them immutable
        try:
            self.true_index = tuple(true_index)
        except TypeError:
            self.true_index = true_index
        try:
            self.pred_index = tuple(pred_index)
        except TypeError:
            self.pred_index = pred_index
        
        self._all_bools()
            

    def __eq__(self, other):
        """Custom comparator. Detections with the same indices are the same."""
        try:
            is_true_same = self.true_index == other.true_index
            is_pred_same = self.pred_index == other.pred_index
            return is_true_same and is_pred_same
        except AttributeError:
            return False

    def __hash__(self):
        """Custom hasher, allow Detections to be hashable."""
        return tuple((self.true_index, self.pred_index)).__hash__()

    def __repr__(self):
        return 'Detection({}, {})'.format(self.true_index, self.pred_index)
    
    def _all_bools(self):
        self._test_correct()
        self._test_gained()
        self._test_missed()
        self._test_split()
        self._test_merge()
        self._test_catastrophe()

    def _test_correct(self):
        self.is_correct = self.true_index is not None and self.pred_index is not None

    def _test_gained(self):
        self.is_gained = self.true_index is None and self.pred_index is not None

    def _test_missed(self):
        self.is_missed = self.true_index is not None and self.pred_index is None

    def _test_split(self):

        if self.is_gained or self.is_missed:
            self.is_split = False
        
        else:
            try:
                is_many_pred = len(self.pred_index) > 1
            except TypeError:
                is_many_pred = False

            try:
                is_single_true = len(tuple(self.true_index)) == 1
            except TypeError:
                is_single_true = isinstance(self.true_index, int)

            self.is_split = is_single_true and is_many_pred

    def _test_merge(self):
        if self.is_gained or self.is_missed:
            self.is_merge = False
        else:

            try:
                is_many_true = len(self.true_index) > 1
            except TypeError:
                is_many_true = False

            try:
                is_single_pred = len(tuple(self.pred_index)) == 1
            except TypeError:
                is_single_pred = isinstance(self.pred_index, int)

            self.is_merge = is_single_pred and is_many_true

    def _test_catastrophe(self):
        if self.is_gained or self.is_missed:
            self.is_catastrophe = False
        
        else:

            try:
                is_many_true = len(self.true_index) > 1
            except TypeError:
                is_many_true = False

            try:
                is_many_pred = len(self.pred_index) > 1
            except TypeError:
                is_many_pred = False

            self.is_catastrophe = is_many_true and is_many_pred


class BaseMetrics(object):  # pylint: disable=useless-object-inheritance

    """Base class for Metrics classes."""

    def __init__(self, y_true, y_pred):
        if y_pred.shape != y_true.shape:
            raise ValueError('Input shapes must match. Shape of prediction '
                             'is: {}.  Shape of y_true is: {}'.format(
                                 y_pred.shape, y_true.shape))

        if not np.issubdtype(y_true.dtype, np.integer):
            warnings.warn('Casting y_true from {} to int'.format(y_true.dtype))
            y_true = y_true.astype('int32')

        if not np.issubdtype(y_pred.dtype, np.integer):
            warnings.warn('Casting y_pred from {} to int'.format(y_pred.dtype))
            y_pred = y_pred.astype('int32')

        self.y_true = y_true
        self.y_pred = y_pred


class PixelMetrics(BaseMetrics):
    """Calculates pixel-based statistics.
    (Dice, Jaccard, Precision, Recall, F-measure)

    Takes in raw prediction and truth data in order to calculate accuracy
    metrics for pixel based classfication. Statistics were chosen according
    to the guidelines presented in Caicedo et al. (2018) Evaluation of Deep
    Learning Strategies for Nucleus Segmentation in Fluorescence Images.
    BioRxiv 335216.

    Args:
        y_true (numpy.array): Binary ground truth annotations for a single
            feature, (batch,x,y)
        y_pred (numpy.array): Binary predictions for a single feature,
            (batch,x,y)

    Raises:
        ValueError: Shapes of y_true and y_pred do not match.

    Warning:
        Comparing labeled to unlabeled data will produce low accuracy scores.
        Make sure to input the same type of data for y_true and y_pred
    """

    def __init__(self, y_true, y_pred):
        super(PixelMetrics, self).__init__(
            y_true=(y_true != 0).astype('int'),
            y_pred=(y_pred != 0).astype('int'))

        self._y_true_sum = np.count_nonzero(self.y_true)
        self._y_pred_sum = np.count_nonzero(self.y_pred)

        # Calculations for IOU
        self._intersection = np.count_nonzero(np.logical_and(self.y_true, self.y_pred))
        self._union = np.count_nonzero(np.logical_or(self.y_true, self.y_pred))

    @property
    def recall(self):
        try:
            _recall = self._intersection / self._y_true_sum
        except ZeroDivisionError:
            _recall = np.nan
        return _recall

    @property
    def precision(self):
        try:
            _precision = self._intersection / self._y_pred_sum
        except ZeroDivisionError:
            _precision = 0
        return _precision

    @property
    def f1(self):
        _recall = self.recall
        _precision = self.precision

        # f1 is nan if recall is nan and no false negatives
        if np.isnan(_recall) and _precision == 0:
            return np.nan

        f_measure = hmean([_recall, _precision])
        # f_measure = (2 * _precision * _recall) / (_precision + _recall)
        return f_measure

    @property
    def dice(self):
        y_sum = self._y_true_sum + self._y_pred_sum
        if y_sum == 0:
            warnings.warn('DICE score is technically 1.0, '
                          'but prediction and truth arrays are empty.')
            return 1.0

        return 2.0 * self._intersection / y_sum

    @property
    def jaccard(self):
        try:
            _jaccard = self._intersection / self._union
        except ZeroDivisionError:
            _jaccard = np.nan
        return _jaccard

    @classmethod
    def to_dict(self):
        return {
            'jaccard': self.jaccard,
            'recall': self.recall,
            'precision': self.precision,
            'f1': self.f1,
            'dice': self.dice,
        }

class ObjectMetrics(BaseMetrics):
    """Classifies object prediction errors as TP, FP, FN, merge or split"""
    
    def __init__(self,
                 y_true,
                 y_pred,
                 cutoff1=0.4,
                 cutoff2=0.1,
                 force_event_links=False,
                 is_3d=False):

        self._validate_inputs(y_true, y_pred, is_3d)
        super(ObjectMetrics, self).__init__(y_true=y_true, y_pred=y_pred)
        
        # Store configuration
        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2
        self.is_3d = is_3d
        self.compute_overlap = compute_overlap_vectorized
        
        # Count objects
        self.n_true = len(np.unique(self.y_true[np.nonzero(self.y_true)]))
        self.n_pred = len(np.unique(self.y_pred[np.nonzero(self.y_pred)]))
        
        # Initialize storage for detections and errors
        self._init_detection_storage()
    
        # Calculate overlap between all pairs of ground truth and predicted objects
        self.iou, self.seg_thresh = self._compute_iou_matrix()
        
        # Optionally boost IoU values for small cells to prevent misclassification
        if force_event_links:
            self.iou_modified = self._boost_small_cell_ious(self.iou)
        else:
            self.iou_modified = self.iou.copy()

        # Use Hungarian algorithm to find optimal 1:1 matches between objects
        matched_pairs = self._match_objects_optimally()
        
        # Record all 1:1 matches as correct detections
        for true_idx, pred_idx in matched_pairs:
            self._add_detection(true_index=int(true_idx), pred_index=int(pred_idx))
        
        # Calculate SEG score from matched pairs
        self.seg_score = self._calculate_seg_score(matched_pairs)

        # Build graph of unmatched objects that overlap
        unmatched_graph = self._build_overlap_graph(matched_pairs)
        
        # Classify graph components as splits, merges, catastrophes, or FP/FN
        self._classify_unmatched_objects(unmatched_graph)

        self.pixel_stats = PixelMetrics(y_true, y_pred)
    
    def _validate_inputs(self, y_true, y_pred, is_3d):
        """Validate input dimensions based on data type."""
        if not is_3d and y_true.ndim not in {2, 3}:
            raise ValueError(
                f'Expected dimensions for y_true (2D data) are 2 (x, y) or 3 (x, y, chan). '
                f'Got ndim: {y_true.ndim}'
            )
        elif is_3d and y_true.ndim != 3:
            raise ValueError(
                f'Expected dimensions for y_true (3D data) is 3 (z, x, y). '
                f'Got ndim: {y_true.ndim}'
            )
    
    def _init_detection_storage(self):
        """Initialize all data structures for tracking detections and errors."""
        # All detected pairs
        self._detections = set()
        
        # Error classifications
        self._splits = set()
        self._merges = set()
        self._catastrophes = set()
        self._gained = set()
        self._missed = set()
        self._correct = set()
        
        # IoU matrices (allocated here, populated in stage 1)
        self.iou = np.zeros((self.n_true, self.n_pred))
        self.seg_thresh = np.zeros((self.n_true, self.n_pred))
    
    def _compute_iou_matrix(self):
        """
        STAGE 1: Compute IoU between all pairs of true and predicted objects.
        
        Returns:
            tuple: (iou_matrix, seg_threshold_matrix)
                - iou_matrix: (n_true, n_pred) intersection over union values
                - seg_threshold_matrix: (n_true, n_pred) binary, 1 if intersection > 0.5 * true_area
        """
        # Early return if either frame is empty
        if self.n_true == 0:
            logging.info('Ground truth frame is empty')
            return self.iou, self.seg_thresh
        
        if self.n_pred == 0:
            logging.info('Prediction frame is empty')
            return self.iou, self.seg_thresh
        
        # Get bounding boxes for efficient overlap detection
        y_true_boxes, y_true_labels = get_box_labels(self.y_true)
        y_pred_boxes, y_pred_labels = get_box_labels(self.y_pred)
        
        if not y_true_boxes.shape[0] or not y_pred_boxes.shape[0]:
            return self.iou, self.seg_thresh
        
        # Find which bounding boxes overlap (cheap computation)
        bbox_overlaps = self.compute_overlap(y_true_boxes, y_pred_boxes)
        ind_true, ind_pred = np.nonzero(bbox_overlaps)
        
        # For overlapping boxes, compute precise pixel-wise IoU
        iou_matrix = np.zeros((self.n_true, self.n_pred))
        seg_matrix = np.zeros((self.n_true, self.n_pred))
        
        for i in range(ind_true.shape[0]):
            true_label = y_true_labels[ind_true[i]]
            pred_label = y_pred_labels[ind_pred[i]]
            
            # Get pixel masks for this pair
            true_mask = (self.y_true == true_label)
            pred_mask = (self.y_pred == pred_label)
            
            # Compute IoU
            intersection = np.count_nonzero(np.logical_and(true_mask, pred_mask))
            union = np.count_nonzero(np.logical_or(true_mask, pred_mask))
            iou_value = intersection / union
            
            # Store in matrix (labels are 1-indexed, matrix is 0-indexed)
            iou_matrix[true_label - 1, pred_label - 1] = iou_value
            
            # Check SEG threshold: intersection > 0.5 * true_area
            true_area = np.count_nonzero(true_mask)
            if intersection > 0.5 * true_area:
                seg_matrix[true_label - 1, pred_label - 1] = 1
        
        return iou_matrix, seg_matrix
    
    def _boost_small_cell_ious(self, iou_matrix):
        """
        STAGE 1B: Modify IoU values to prevent small cells from being missed.
        
        When a small cell has low IoU (because it's small) but is mostly contained
        within a larger cell, boost its IoU so it gets properly matched.
        
        Args:
            iou_matrix: Original IoU matrix
            
        Returns:
            np.array: Modified IoU matrix with boosted values for small cells
        """
        # Find pairs with low IoU but potential overlap
        true_idx, pred_idx = np.nonzero(
            np.logical_and(iou_matrix > 0, iou_matrix < 1 - self.cutoff1)
        )
        
        iou_modified = iou_matrix.copy()
        
        for i in range(len(true_idx)):
            t_idx, p_idx = true_idx[i], pred_idx[i]
            true_label, pred_label = t_idx + 1, p_idx + 1
            
            # Get masks
            true_mask = (self.y_true == true_label)
            pred_mask = (self.y_pred == pred_label)
            
            # Calculate containment fractions
            true_in_pred = (
                np.count_nonzero(self.y_true[pred_mask] == true_label) / 
                np.sum(true_mask)
            )
            pred_in_true = (
                np.count_nonzero(self.y_pred[true_mask] == pred_label) / 
                np.sum(pred_mask)
            )
            
            max_containment = np.max([true_in_pred, pred_in_true])
            
            # If small cell is >50% contained, boost its IoU
            if iou_matrix[t_idx, p_idx] <= self.cutoff1 and max_containment > 0.5:
                iou_modified[t_idx, p_idx] = self.cutoff2
                
                # Also reduce IoU of the large cell to prevent incorrect matches
                if true_in_pred > 0.5:
                    # Find other high-IoU predictions for this pred cell
                    high_iou_idx = np.nonzero(
                        iou_matrix[:, p_idx] >= 1 - self.cutoff1
                    )[0]
                    iou_modified[high_iou_idx, p_idx] = 1 - self.cutoff1 - 0.01
                
                if pred_in_true > 0.5:
                    # Find other high-IoU predictions for this true cell
                    high_iou_idx = np.nonzero(
                        iou_matrix[t_idx, :] >= 1 - self.cutoff1
                    )[0]
                    iou_modified[t_idx, high_iou_idx] = 1 - self.cutoff1 - 0.01
        
        return iou_modified
    
    def _match_objects_optimally(self):
        """
        STAGE 2: Use Hungarian algorithm to find optimal 1:1 object matches.
        
        Constructs a cost matrix from IoU values and finds the assignment that
        minimizes total cost (maximizes total IoU).
        
        Returns:
            list of tuples: [(true_idx, pred_idx), ...] for matched pairs
        """
        # Build cost matrix (convert from IoU matrix)
        n_obj = self.n_true + self.n_pred
        cost_matrix = np.ones((n_obj, n_obj))
        
        # Top-left: cost of matching true to pred (1 - IoU)
        cost = 1 - self.iou_modified
        cost_matrix[:self.n_true, :self.n_pred] = cost
        
        # Bottom-right: cost of matching pred to true (transpose)
        cost_matrix[n_obj - self.n_pred:, n_obj - self.n_true:] = cost.T
        
        # Diagonal corners: cost of NOT matching (leaving unmatched)
        # Lower-left corner: cost for pred to be unmatched
        bl_corner = (
            self.cutoff1 * np.eye(self.n_pred) + 
            np.ones((self.n_pred, self.n_pred)) - 
            np.eye(self.n_pred)
        )
        cost_matrix[n_obj - self.n_pred:, :self.n_pred] = bl_corner
        
        # Upper-right corner: cost for true to be unmatched
        tr_corner = (
            self.cutoff1 * np.eye(self.n_true) + 
            np.ones((self.n_true, self.n_true)) - 
            np.eye(self.n_true)
        )
        cost_matrix[:self.n_true, n_obj - self.n_true:] = tr_corner
        
        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Extract only the true-to-pred matches (top-left quadrant)
        matched_pairs = []
        for r, c in zip(row_ind, col_ind):
            if r < self.n_true and c < self.n_pred:
                matched_pairs.append((r, c))
        
        return matched_pairs
    
    def _calculate_seg_score(self, matched_pairs):
        """Calculate SEG score from matched pairs (mean IoU where intersection > 0.5 * true area)."""
        if not matched_pairs:
            return np.nan
        
        # Mask IoU values where seg_thresh is not met
        iou_masked = np.where(self.seg_thresh == 0, self.iou, np.nan)
        
        # Extract IoU values for matched pairs only
        iou_values = [iou_masked[t_idx, p_idx] for t_idx, p_idx in matched_pairs]
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return np.nanmean(iou_values)
    
    def _build_overlap_graph(self, matched_pairs):
        """
        STAGE 3A: Build graph connecting unmatched objects that overlap.
        
        Args:
            matched_pairs: List of (true_idx, pred_idx) tuples already matched
            
        Returns:
            networkx.Graph: Graph where nodes are unmatched objects and edges
                          represent overlaps above cutoff2 threshold
        """
        # Identify which objects were NOT matched
        matched_true = set(t for t, p in matched_pairs)
        matched_pred = set(p for t, p in matched_pairs)
        
        unmatched_true = [i for i in range(self.n_true) if i not in matched_true]
        unmatched_pred = [i for i in range(self.n_pred) if i not in matched_pred]
        
        # Build graph
        G = nx.Graph()
        
        # Add edges for overlapping unmatched objects
        for t_idx in unmatched_true:
            for p_idx in unmatched_pred:
                if self.iou_modified[t_idx, p_idx] >= self.cutoff2:
                    G.add_edge(f'true_{t_idx}', f'pred_{p_idx}')
        
        # Ensure all unmatched objects are in graph (even isolated ones)
        for t_idx in unmatched_true:
            G.add_node(f'true_{t_idx}')
        for p_idx in unmatched_pred:
            G.add_node(f'pred_{p_idx}')
        
        return G
    
    def _classify_unmatched_objects(self, G):
        """
        STAGE 3B: Classify connected components in overlap graph.
        
        Graph classification rules:
        - Isolated nodes: FP (pred) or FN (true)
        - 1-1 connections: TP (should have been matched earlier)
        - 1-many: Split (1 true → many pred) or Merge (many true → 1 pred)
        - many-many: Catastrophe
        """
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            
            # Find the node with highest degree (most connections)
            max_degree = max(dict(subgraph.degree).values()) if subgraph.nodes else 0
            
            # Separate true and pred nodes
            true_indices = []
            pred_indices = []
            
            for node in subgraph.nodes:
                node_type, idx_str = node.split('_')
                idx = int(idx_str) + 1  # Convert back to 1-indexed labels
                
                if node_type == 'true':
                    if max_degree > 1:
                        true_indices.append(idx)
                    else:
                        # Isolated or 1-1, check if already added
                        self._add_detection(true_index=idx)
                else:  # node_type == 'pred'
                    if max_degree > 1:
                        pred_indices.append(idx)
                    else:
                        # Isolated or 1-1
                        self._add_detection(pred_index=idx)
            
            # Add many-to-many detections (splits, merges, catastrophes)
            if true_indices or pred_indices:
                self._add_detection(
                    true_index=tuple(true_indices) if true_indices else None,
                    pred_index=tuple(pred_indices) if pred_indices else None,
                )
    
    def _add_detection(self, true_index=None, pred_index=None):
        """Record a detection and classify its error type."""
        detection = Detection(true_index=true_index, pred_index=pred_index)
        self._detections.add(detection)
        
        # Classify detection type
        if detection.is_correct:
            self._correct.add(detection)
        if detection.is_gained:
            self._gained.add(detection)
        if detection.is_missed:
            self._missed.add(detection)
        if detection.is_split:
            self._splits.add(detection)
        if detection.is_merge:
            self._merges.add(detection)
        if detection.is_catastrophe:
            self._catastrophes.add(detection)

    @property
    def correct_detections(self):
        return len(self._correct)

    @property
    def missed_detections(self):
        return len(self._missed)

    @property
    def gained_detections(self):
        return len(self._gained)

    @property
    def splits(self):
        return len(self._splits)

    @property
    def merges(self):
        return len(self._merges)

    @property
    def catastrophes(self):
        return len(self._catastrophes)

    @property
    def gained_det_from_split(self):
        gained_dets = 0
        for det in self._splits:
            true_idx = cast_to_tuple(det.true_index)
            pred_idx = cast_to_tuple(det.pred_index)
            gained_dets += len(true_idx) + len(pred_idx) - 2
        return gained_dets

    @property
    def missed_det_from_merge(self):
        missed_dets = 0
        for det in self._merges:
            true_idx = cast_to_tuple(det.true_index)
            pred_idx = cast_to_tuple(det.pred_index)
            missed_dets += len(true_idx) + len(pred_idx) - 2
        return missed_dets

    @property
    def true_det_in_catastrophe(self):
        return sum([len(d.true_index) for d in self._catastrophes])

    @property
    def pred_det_in_catastrophe(self):
        return sum([len(d.pred_index) for d in self._catastrophes])

    @property
    def split_props(self):
        return self._get_props('splits')

    @property
    def merge_props(self):
        return self._get_props('merges')

    @property
    def missed_props(self):
        return self._get_props('missed')

    @property
    def gained_props(self):
        return self._get_props('gained')

    @property
    def recall(self):
        try:
            recall = self.correct_detections / self.n_true
        except ZeroDivisionError:
            recall = 0
        return recall

    @property
    def precision(self):
        try:
            precision = self.correct_detections / self.n_pred
        except ZeroDivisionError:
            precision = 0
        return precision

    @property
    def f1(self):
        return hmean([self.recall, self.precision])

    @property
    def jaccard(self):
        return self.pixel_stats.jaccard

    @property
    def dice(self):
        return self.pixel_stats.dice

    def to_dict(self):
        """Return a dictionary representation of the calclulated metrics."""
        return {
            'n_pred': self.n_pred,
            'n_true': self.n_true,
            'correct_detections': self.correct_detections,
            'missed_detections': self.missed_detections,
            'gained_detections': self.gained_detections,
            'missed_det_from_merge': self.missed_det_from_merge,
            'gained_det_from_split': self.gained_det_from_split,
            'true_det_in_catastrophe': self.true_det_in_catastrophe,
            'pred_det_in_catastrophe': self.pred_det_in_catastrophe,
            'merge': self.merges,
            'split': self.splits,
            'catastrophe': self.catastrophes,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'seg': self.seg_score,
            'jaccard': self.jaccard,
            'dice': self.dice,
        }

class Metrics(object):
    """Class to calculate and save various segmentation metrics.

    Args:
        model_name (str): Name of the model which determines output file names
        outdir (:obj:`str`, optional): Directory to save json file, default ''
        cutoff1 (:obj:`float`, optional): Threshold for overlap in cost matrix,
            smaller values are more conservative, default 0.4
        cutoff2 (:obj:`float`, optional): Threshold for overlap in unassigned
            cells, smaller values are better, default 0.1
        pixel_threshold (:obj:`float`, optional): Threshold for converting
            predictions to binary
        ndigits (:obj:`int`, optional): Sets number of digits for rounding,
            default 4
        feature_key (:obj:`list`, optional): List of strings, feature names
        json_notes (:obj:`str`, optional): Str providing any additional
            information about the model
        force_event_links(:obj:`bool`, optional): Flag that determines whether to modify IOU
            calculation so that merge or split events with cells of very different sizes are
            never misclassified as misses/gains.
        is_3d(:obj:`bool`, optional): Flag that determines whether or not the input data
            should be treated as 3-dimensional.

    Examples:
        >>> from deepcell import metrics
        >>> m = metrics.Metrics('model_name')
        >>> all_metrics = m.run_all(y_true, y_pred)
        >>> m.save_to_json(all_metrics)
    """
    def __init__(self, model_name,
                 outdir='',
                 cutoff1=0.4,
                 cutoff2=0.1,
                 pixel_threshold=0.5,
                 ndigits=4,
                 crop_size=None,
                 feature_key=[],
                 json_notes='',
                 force_event_links=False,
                 is_3d=False,
                 **kwargs):
        self.model_name = model_name
        self.outdir = outdir
        self.cutoff1 = cutoff1
        self.cutoff2 = cutoff2
        self.pixel_threshold = pixel_threshold
        self.ndigits = ndigits
        self.crop_size = crop_size
        self.feature_key = feature_key
        self.json_notes = json_notes
        self.force_event_links = force_event_links
        self.is_3d = is_3d

        # Initialize output list to collect stats
        self.object_metrics = []
        self.pixel_metrics = []

    def df_to_dict(self, df, stat_type='pixel'):
        """Output pandas df as a list of dictionary objects

        Args:
            df (pandas.DataFrame): Dataframe of statistics for each channel
            stat_type (str): Category of statistic.

        Returns:
            list: List of dictionaries
        """

        # Initialize output dictionary
        L = []

        # Write out average statistics
        for k, v in df.mean().items():
            L.append(dict(
                name=k,
                value=v,
                feature='average',
                stat_type=stat_type,
            ))

        # Save individual stats to list
        for i, row in df.iterrows():
            for k, v in row.items():
                L.append(dict(
                    name=k,
                    value=v,
                    feature=i,
                    stat_type=stat_type,
                ))

        return L

    def calc_pixel_stats(self, y_true, y_pred, axis=-1):
        """Calculate pixel statistics for each feature.

        ``y_true`` should have the appropriate transform applied to match
        ``y_pred``. Each channel is converted to binary using the threshold
        ``pixel_threshold`` prior to calculation of accuracy metrics.

        Args:
            y_true (numpy.array): Ground truth annotations after transform
            y_pred (numpy.array): Model predictions without labeling

        Returns:
            list: list of dictionaries with each stat being a key.

        Raises:
            ValueError: If y_true and y_pred are not the same shape
        """
        n_features = y_pred.shape[axis]

        pixel_metrics = []

        slc = [slice(None)] * y_pred.ndim
        for i in range(n_features):
            slc[axis] = slice(i, i + 1)
            yt = y_true[tuple(slc)] > self.pixel_threshold
            yp = y_pred[tuple(slc)] > self.pixel_threshold
            pm = PixelMetrics(yt, yp)
            pixel_metrics.append(pm.to_dict())

        pixel_df = pd.DataFrame.from_records(pixel_metrics)

        # Calculate confusion matrix
        cm = PixelMetrics.get_confusion_matrix(y_true, y_pred, axis=axis)

        print('\n____________Pixel-based statistics____________\n')
        print(pixel_df)
        print('\nConfusion Matrix')
        print(cm)

        output = self.df_to_dict(pixel_df)

        output.append(dict(
            name='confusion_matrix',
            value=cm.tolist(),
            feature='all',
            stat_type='pixel'
        ))
        return output

    def calc_object_stats(self, y_true, y_pred, progbar=True):
        """Calculate object statistics and save to output

        Loops over each frame in the zeroth dimension, which should pass in
        a series of 2D arrays for analysis. 'metrics.split_stack' can be
        used to appropriately reshape the input array if necessary

        Args:
            y_true (numpy.array): Labeled ground truth annotations
            y_pred (numpy.array): Labeled prediction mask
            progbar (bool): Whether to show the progress tqdm progress bar

        Returns:
            list: list of dictionaries with each stat being a key.

        Raises:
            ValueError: If y_true and y_pred are not the same shape
            ValueError: If data_type is 2D, if input shape does not have ndim 3 or 4
            ValueError: If data_type is 3D, if input shape does not have ndim 4
        """
        if y_pred.shape != y_true.shape:
            raise ValueError('Input shapes need to match. Shape of prediction '
                             'is: {}.  Shape of y_true is: {}'.format(
                                 y_pred.shape, y_true.shape))

        # If 2D, dimensions can be 3 or 4 (with or without channel dimension)
        if not self.is_3d:
            if y_true.ndim not in {3, 4}:
                raise ValueError('Expected dimensions for y_true (2D data) are 3 or 4.'
                                 'Accepts: (batch, x, y), or (batch, x, y, chan)'
                                 'Got ndim: {}'.format(y_true.ndim))

        # If 3D, inputs must have 4 dimensions (batch, z, x, y) - cannot have channel dimension or
        # _classify_graph breaks, as it expects input to be 2D or 3D
        # TODO - add compatibility for multi-channel 3D-data
        else:
            if y_true.ndim != 4:
                raise ValueError('Expected dimensions for y_true (3D data) is 4. '
                                 'Required format is: (batch, z, x, y) '
                                 'Got ndim: {}'.format(y_true.ndim))

        all_object_metrics = []  # store all calculated metrics
        is_batch_relabeled = False  # used to warn if batches were relabeled

        for i in tqdm(range(y_true.shape[0]), disable=not progbar):
            # check if labels aren't sequential, raise warning on first occurence if so
            true_batch, pred_batch = y_true[i], y_pred[i]
            true_batch_relabel, _, _ = relabel_sequential(true_batch)
            pred_batch_relabel, _, _ = relabel_sequential(pred_batch)

            # check if segmentations were relabeled
            if not is_batch_relabeled:  # only one True is required
                is_batch_relabeled = not (
                    np.array_equal(true_batch, true_batch_relabel)
                    and np.array_equal(pred_batch, pred_batch_relabel)
                )

            o = ObjectMetrics(
                true_batch_relabel,
                pred_batch_relabel,
                cutoff1=self.cutoff1,
                cutoff2=self.cutoff2,
                force_event_links=self.force_event_links,
                is_3d=self.is_3d)

            all_object_metrics.append(o)

        # print the object report
        object_metrics = pd.DataFrame.from_records([
            o.to_dict() for o in all_object_metrics
        ])
        self.print_object_report(object_metrics)
        return object_metrics

    def summarize_object_metrics_df(self, df):
        correct_detections = int(df['correct_detections'].sum())
        n_true = int(df['n_true'].sum())
        n_pred = int(df['n_pred'].sum())

        _round = lambda x: round(x, self.ndigits)

        seg = df['seg'].mean()
        jaccard = df['jaccard'].mean()

        try:
            recall = correct_detections / n_true
        except ZeroDivisionError:
            recall = np.nan
        try:
            precision = correct_detections / n_pred
        except ZeroDivisionError:
            precision = 0

        errors = [
            'gained_detections',
            'missed_detections',
            'split',
            'merge',
            'catastrophe',
        ]

        bad_detections = [
            'gained_det_from_split',
            'missed_det_from_merge',
            'true_det_in_catastrophe',
            'pred_det_in_catastrophe',
        ]

        summary = {
            'correct_detections': correct_detections,
            'n_true': n_true,
            'n_pred': n_pred,
            'recall': _round(recall),
            'precision': _round(precision),
            'seg': _round(seg),
            'jaccard': _round(jaccard),
            'total_errors': 0,
        }
        # update bad detections
        for k in bad_detections:
            summary[k] = int(df[k].sum())
        # update error counts
        for k in errors:
            count = int(df[k].sum())
            summary[k] = count
            summary['total_errors'] += count
        return summary

    def print_object_report(self, object_metrics):
        """Print neat report of object based statistics

        Args:
            object_metrics (pd.DataFrame): DataFrame of all calculated metrics
        """
        summary = self.summarize_object_metrics_df(object_metrics)
        errors = [
            'gained_detections',
            'missed_detections',
            'split',
            'merge',
            'catastrophe'
        ]

        bad_detections = [
            'gained_det_from_split',
            'missed_det_from_merge',
            'true_det_in_catastrophe',
            'pred_det_in_catastrophe',
        ]

        # print('\n____________Object-based statistics____________\n')
        # print('Number of true cells:\t\t', summary['n_true'])
        # print('Number of predicted cells:\t', summary['n_pred'])

        # print('\nCorrect detections:  {}\tRecall: {}%'.format(
        #     summary['correct_detections'], summary['recall']))

        # print('Incorrect detections: {}\tPrecision: {}%'.format(
        #     summary['n_pred'] - summary['correct_detections'],
        #     summary['precision']))

        # print('\n')
        # for k in errors:
        #     v = summary[k]
        #     name = k.replace('_', ' ').capitalize()
        #     if not name.endswith('s'):
        #         name += 's'

        #     try:
        #         err_fraction = v / summary['total_errors']
        #     except ZeroDivisionError:
        #         err_fraction = 0

        #     print('{name}: {val}{tab} Perc Error {percent}%'.format(
        #         name=name, val=v,
        #         percent=round(100 * err_fraction, self.ndigits),
        #         tab='\t' * (1 if ' ' in name else 2)))

        # for k in bad_detections:
        #     name = k.replace('_', ' ').capitalize().replace(' det ', ' detections')
        #     print('{name}: {val}'.format(name=name, val=summary[k]))

        # print('SEG:', round(summary['seg'], self.ndigits), '\n')

        # print('Average Pixel IOU (Jaccard Index):',
        #       round(summary['jaccard'], self.ndigits), '\n')

    def run_all(self, y_true, y_pred, axis=-1):
        object_metrics = self.calc_object_stats(y_true, y_pred)
        pixel_metrics = self.calc_pixel_stats(y_true, y_pred, axis=axis)

        object_list = self.df_to_dict(object_metrics, stat_type='object')
        all_output = object_list + pixel_metrics
        self.save_to_json(all_output)

    def save_to_json(self, L):
        """Save list of dictionaries to json file with file metadata

        Args:
            L (list): List of metric dictionaries
        """
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        outname = os.path.join(
            self.outdir, '{}_{}.json'.format(self.model_name, todays_date))

        # Configure final output
        D = {}

        # Record metadata
        D['metadata'] = dict(
            model_name=self.model_name,
            date=todays_date,
            notes=self.json_notes
        )

        # Record metrics
        D['metrics'] = L

        with open(outname, 'w') as outfile:
            json.dump(D, outfile)

        logging.info('Saved to {}'.format(outname))


if __name__ == "__main__":
    import numpy as np
    from scipy.ndimage import label
    
    print("=" * 60)
    print("Testing Refactored ObjectMetrics")
    print("=" * 60)
    
    # ============================================================
    # Create synthetic test data with known error types
    # ============================================================
    
    # Create a 100x100 image
    y_true = np.zeros((100, 100), dtype=np.int32)
    y_pred = np.zeros((100, 100), dtype=np.int32)
    
    # Cell 1: Perfect match (True Positive)
    y_true[10:20, 10:20] = 1
    y_pred[10:20, 10:20] = 1
    
    # Cell 2: Missed detection (False Negative)
    y_true[30:40, 10:20] = 2
    # No corresponding prediction
    
    # Cell 3: Gained detection (False Positive)
    # No corresponding ground truth
    y_pred[50:60, 10:20] = 2
    
    # Cell 4: Split (1 true → 2 pred)
    y_true[10:30, 30:50] = 3
    y_pred[10:20, 30:50] = 3  # Top half
    y_pred[20:30, 30:50] = 4  # Bottom half
    
    # Cell 5: Merge (2 true → 1 pred)
    y_true[40:50, 30:40] = 4
    y_true[50:60, 30:40] = 5
    y_pred[40:60, 30:40] = 5  # Combined
    
    # Cell 6: Good match with slight offset
    y_true[70:85, 70:85] = 6
    y_pred[72:87, 72:87] = 6
    
    print("\nTest Data Created:")
    print(f"  Ground truth objects: {len(np.unique(y_true)) - 1}")  # -1 for background
    print(f"  Predicted objects: {len(np.unique(y_pred)) - 1}")
    print("\nExpected errors:")
    print("  - 2 True Positives (cells 1 and 6)")
    print("  - 1 False Negative (cell 2)")
    print("  - 1 False Positive (cell 3)")
    print("  - 1 Split (cell 4: 1 true → 2 pred)")
    print("  - 1 Merge (cells 4-5: 2 true → 1 pred)")
    
    # ============================================================
    # Run the metrics (this will execute our refactored __init__)
    # ============================================================
    print("\n" + "=" * 60)
    print("Running ObjectMetrics...")
    print("=" * 60)
    
    # Need to add a batch dimension for ObjectMetrics
    y_true_batch = y_true[np.newaxis, ...]
    y_pred_batch = y_pred[np.newaxis, ...]
    
    try:
        metrics = ObjectMetrics(
            y_true=y_true_batch,
            y_pred=y_pred_batch,
            cutoff1=0.4,
            cutoff2=0.1,
            force_event_links=False,
            is_3d=False
        )
        
        print("\n✓ ObjectMetrics initialized successfully!")
        
        # ============================================================
        # Display results
        # ============================================================
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        print(f"\nObjects detected:")
        print(f"  True objects (n_true): {metrics.n_true}")
        print(f"  Predicted objects (n_pred): {metrics.n_pred}")
        
        print(f"\nCorrect detections: {metrics.correct_detections}")
        print(f"Missed detections: {metrics.missed_detections}")
        print(f"Gained detections: {metrics.gained_detections}")
        print(f"Splits: {metrics.splits}")
        print(f"Merges: {metrics.merges}")
        print(f"Catastrophes: {metrics.catastrophes}")
        
        print(f"\nPerformance metrics:")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall: {metrics.recall:.3f}")
        print(f"  F1 Score: {metrics.f1:.3f}")
        print(f"  SEG Score: {metrics.seg_score:.3f}")
        print(f"  Jaccard (IoU): {metrics.jaccard:.3f}")
        print(f"  Dice: {metrics.dice:.3f}")
        
        # ============================================================
        # Verify IoU matrix was populated
        # ============================================================
        print(f"\nIoU Matrix shape: {metrics.iou.shape}")
        print(f"Non-zero IoU values: {np.count_nonzero(metrics.iou)}")
        print(f"Max IoU value: {np.max(metrics.iou):.3f}")
        
        # Show the IoU matrix for matched objects
        print("\nIoU Matrix (ground truth × predicted):")
        print("(showing values > 0)")
        for i in range(metrics.iou.shape[0]):
            for j in range(metrics.iou.shape[1]):
                if metrics.iou[i, j] > 0:
                    print(f"  True[{i+1}] × Pred[{j+1}]: {metrics.iou[i, j]:.3f}")
        
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during metrics calculation:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
    # ============================================================
    # Visual representation
    # ============================================================
    print("\nGround Truth Layout:")
    print("  [1: Perfect]  [2: Missed]")
    print("  [3: Split source]")
    print("  [4,5: Merge sources]")
    print("  [6: Good match]")
    
    print("\nPrediction Layout:")
    print("  [1: Perfect]  [X: No pred]  [2: Gained]")
    print("  [3,4: Split results]")
    print("  [5: Merge result]")
    print("  [6: Good match]")


