import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_mesmer.transforms import transform_masks
from torch_mesmer.augmentation import MultiTransform
from torch_mesmer.utils import histogram_normalization, percentile_threshold

# Targets, normalisation and quality control of the published Mesmer training recipe.
IN_TRANSFORMS = ('inner-distance', 'pixelwise')
TRANSFORMS_KWARGS = {
    'pixelwise': {'dilation_radius': 1},
    'inner-distance': {'erosion_width': 1, 'alpha': 'auto', 'beta': 1},
}
CLAHE_KERNEL_SIZE = 128   # offline multiplex_preprocess on the full image
ZOOM_MIN = 0.7            # zoom range (0.7, 1/0.7)
MIN_OBJECT_SIZE = 15      # paper QC: drop objects smaller than this (pixels)
MIN_OBJECTS = 20          # paper QC: drop images with fewer whole-cell objects
WHOLE_CELL_CHANNEL = 1    # label channel the object count uses (the zarrs are nuclear first)


def target_layout(in_transforms, transforms_kwargs=None):
    """Channels and interpolation mode of each target, for one compartment.

    Returns:
        list of ``(n_channels, mode)``, one per entry of ``in_transforms``.
        Continuous maps are resized bilinearly; one-hot maps with nearest.
    """
    transforms_kwargs = transforms_kwargs or {}
    layout = []
    for transform in in_transforms:
        key = transform.lower().replace('_', '-')
        if key in ('inner-distance', 'outer-distance'):
            layout.append((1, 'bilinear'))
        elif key == 'pixelwise':
            separate = transforms_kwargs.get(transform, {}).get('separate_edge_classes', False)
            layout.append((4 if separate else 3, 'nearest'))
        elif key == 'fgbg':
            layout.append((2, 'nearest'))
        else:
            raise ValueError(f'`{transform}` is not a valid transform')
    return layout


def remove_small_objects(labels, min_size):
    """Zero out every label with fewer than ``min_size`` pixels.

    Same as ``skimage.morphology.remove_small_objects`` on a label image: sizes
    are counted per label value, not per connected component.
    """
    labels = np.asarray(labels).astype(np.int64)
    if min_size <= 0 or not labels.any():
        return labels
    sizes = np.bincount(labels.ravel())
    too_small = sizes < min_size
    too_small[0] = False
    labels[too_small[labels]] = 0
    return labels


def count_objects(labels, min_size=0):
    """Number of labels with at least ``min_size`` pixels (background excluded)."""
    labels = np.asarray(labels).astype(np.int64)
    sizes = np.bincount(labels.ravel())[1:]
    return int(np.count_nonzero(sizes >= max(min_size, 1)))


def qc_keep_indices(y, min_objects=MIN_OBJECTS, min_object_size=MIN_OBJECT_SIZE, channel=WHOLE_CELL_CHANNEL):
    """Indices of the images that pass the paper's QC.

    An image passes if, after removing objects smaller than ``min_object_size``,
    label channel ``channel`` still has at least ``min_objects`` objects
    (``dataset_builder.py:637-679`` in the paper's figures repo). Images are read
    one at a time, so a zarr array is never loaded whole.
    """
    keep = []
    for i in range(y.shape[0]):
        if count_objects(y[i, channel], min_object_size) >= min_objects:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


class SegmentationDataset(Dataset):
    """Images and their training targets, at native resolution.

    ``X`` and ``y`` are ``(N, C, H, W)`` arrays (NumPy or zarr; zarr is read one
    image at a time). Each item is normalised (99.9th-percentile clip, then
    CLAHE), its labels are turned into targets, and for ``dataset_type='train'``
    one random augmentation is applied to the image and every target.

    Target channels are ordered compartment by compartment, then transform by
    transform: for the default ``in_transforms`` and two label channels that is
    ``[c0 inner distance, c0 edge, c0 interior, c0 background, c1 ...]``.

    Args:
        X, y: images and integer label images.
        dataset_type: ``'train'`` (augmented) or ``'val'`` (centre crop only).
        in_transforms: target transforms applied to every label channel.
        transforms_kwargs: keyword arguments for each transform, keyed by name.
        crop_size: output side.
        rotation_range: augmentation rotation limit, degrees.
        zoom_min: augmentation zoom range is ``(zoom_min, 1/zoom_min)`` on each axis.
        kernel_size: CLAHE kernel size.
        min_object_size: objects smaller than this are removed from every label
            channel before the targets are computed. 0 disables it.
        min_objects: images with fewer objects than this in ``qc_channel`` (after
            small-object removal) are left out. 0 disables it.
        qc_channel: label channel ``min_objects`` counts in.
        semantic_heads: optional; if given, it must match the target channels
            implied by ``in_transforms`` (a guard against a model/data mismatch).
    """

    def __init__(self,
                 X,
                 y,
                 dataset_type='train',
                 in_transforms=IN_TRANSFORMS,
                 transforms_kwargs=None,
                 crop_size=256,
                 rotation_range=180,
                 zoom_min=ZOOM_MIN,
                 kernel_size=CLAHE_KERNEL_SIZE,
                 min_object_size=0,
                 min_objects=0,
                 qc_channel=WHOLE_CELL_CHANNEL,
                 semantic_heads=None,
                 ):

        if X.shape[0] != y.shape[0]:
            raise ValueError(f'X has {X.shape[0]} images but y has {y.shape[0]}')
        if X.ndim != 4 or y.ndim != 4:
            raise ValueError(f'X and y must be (N, C, H, W); got {X.shape} and {y.shape}')

        self.X = X
        self.y = y
        self.dataset_type = dataset_type
        self.in_transforms = list(in_transforms)
        self.transforms_kwargs = TRANSFORMS_KWARGS if transforms_kwargs is None else transforms_kwargs
        self.crop_size = crop_size
        self.rotation_range = rotation_range
        self.zoom_min = zoom_min
        self.kernel_size = kernel_size
        self.min_object_size = min_object_size

        # The augmentation sees one stacked tensor: the image channels, then for
        # each compartment the targets of each transform. It needs one
        # interpolation mode per channel, so each target's mode is repeated
        # once per channel of that target.
        layout = target_layout(self.in_transforms, self.transforms_kwargs) * y.shape[1]
        self.head_channels = [n for n, _ in layout]
        self.interpolation = ['bilinear'] * X.shape[1]
        for n_channels, mode in layout:
            self.interpolation += [mode] * n_channels

        if semantic_heads is not None and list(semantic_heads) != self.head_channels:
            raise ValueError(
                f'semantic_heads={list(semantic_heads)}, but {self.in_transforms} on {y.shape[1]} '
                f'label channels gives heads {self.head_channels}')

        if min_objects > 0:
            self.indices = qc_keep_indices(y, min_objects, min_object_size, qc_channel)
        else:
            self.indices = np.arange(X.shape[0])

    def _normalize(self, image):
        """Clip at the 99.9th percentile, then apply CLAHE, channel by channel.

        Args:
            image: ``(C, H, W)``.

        Returns:
            ``(C, H, W)``, float32. The helpers work on batches, so a batch
            axis is added for them and removed again.
        """
        image = np.expand_dims(image, 0)

        assert len(image.shape) == 4, 'add batch dimension'
        image = percentile_threshold(image, percentile=99.9)
        image = histogram_normalization(image, kernel_size=self.kernel_size)

        return image.squeeze(axis=0)

    def _transform_labels(self, y):
        """Turn one image's label channels into its training targets.

        Objects smaller than ``min_object_size`` are removed first.

        Args:
            y: integer labels, ``(compartments, H, W)``.

        Returns:
            ``(sum(head_channels), H, W)``: compartment by compartment, the
            target of each transform in ``in_transforms`` (see the class
            docstring for the order).
        """
        y_semantic = []

        for compartment in range(y.shape[0]):
            labels = y[compartment]
            if self.min_object_size > 0:
                labels = remove_small_objects(labels, self.min_object_size)
            labels = np.expand_dims(labels, 0)

            for transform in self.in_transforms:
                transform_kwargs = self.transforms_kwargs.get(transform, dict())
                y_transform = transform_masks(labels, transform, unbatched=True, **transform_kwargs)
                y_semantic.append(y_transform.squeeze(axis=0))

        return np.concatenate(y_semantic, axis=0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Load, normalise, build the targets of and augment one sample.

        Returns:
            ``(image, targets)``, float32 tensors of ``(C_image, crop_size,
            crop_size)`` and ``(sum(head_channels), crop_size, crop_size)``.
        """
        i = int(self.indices[idx])
        x = np.asarray(self.X[i])
        y = np.asarray(self.y[i])

        x = self._normalize(x)
        y = self._transform_labels(y)

        # Stack x and y along the channel dimension, so one draw of the
        # augmentation moves them together.
        combined = torch.cat([torch.from_numpy(x).float(), torch.from_numpy(y).float()], dim=0)

        transform = MultiTransform(
            combined.shape[1:],
            dataset_type=self.dataset_type,
            crop_size=self.crop_size,
            rotation_range=self.rotation_range,
            zoom_min=self.zoom_min,
        )
        combined = transform(combined, self.interpolation)

        # Split back into x and y
        n_x = x.shape[0]
        return combined[:n_x], combined[n_x:]


def create_data_loaders(
    train,
    val,
    crop_size=256,
    zoom_min=ZOOM_MIN,
    rotation_range=180,
    batch_size=16,
    num_workers=4,
    in_transforms=IN_TRANSFORMS,
    transforms_kwargs=None,
    kernel_size=CLAHE_KERNEL_SIZE,
    min_object_size=MIN_OBJECT_SIZE,
    min_objects=MIN_OBJECTS,
    semantic_heads=None,
    seed=0,
):
    """Build the train and val loaders.

    The paper's QC (``min_object_size``, ``min_objects``) is applied to train
    only; val stays exactly as provided. Both loaders drop the last partial
    batch, as the published recipe does (it runs ``n // batch_size`` steps per
    epoch). Train is shuffled and augmented; val is neither.

    Args:
        train, val: mappings with ``'X'`` (images) and ``'y'`` (integer
            labels), both ``(N, C, H, W)``, channels first, e.g. an open zarr
            group. Either can be ``None``, and then so is its loader.
        crop_size, zoom_min, rotation_range, in_transforms, transforms_kwargs,
            kernel_size: passed to ``SegmentationDataset``.
        batch_size: samples per batch.
        num_workers: DataLoader worker processes.
        min_object_size, min_objects: train-only QC, see ``SegmentationDataset``.
        semantic_heads: optional check of the target channels against the
            model's heads, see ``SegmentationDataset``.
        seed: seeds the train shuffle order and, through it, each worker's
            augmentation draws (PyTorch seeds ``torch``, ``random`` and
            ``numpy`` in every worker from the loader's generator).

    Returns:
        ``(train_loader, val_loader)``. Each batch is ``(images, targets)``,
        float32 tensors of ``(batch_size, C_image, crop_size, crop_size)`` and
        ``(batch_size, sum(semantic_heads), crop_size, crop_size)``.
    """
    common = dict(
        in_transforms=in_transforms,
        transforms_kwargs=transforms_kwargs,
        crop_size=crop_size,
        rotation_range=rotation_range,
        zoom_min=zoom_min,
        kernel_size=kernel_size,
        semantic_heads=semantic_heads,
    )

    dataloader = None
    valloader = None

    if train is not None:

        train_dataset = SegmentationDataset(
            train['X'],
            train['y'],
            dataset_type='train',
            min_object_size=min_object_size,
            min_objects=min_objects,
            **common)

        generator = torch.Generator()
        generator.manual_seed(seed)

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=True, drop_last=True, generator=generator)

    if val is not None:
        val_dataset = SegmentationDataset(
            val['X'],
            val['y'],
            dataset_type='val',
            **common)

        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                               pin_memory=True, drop_last=True)

    return dataloader, valloader
