import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_mesmer.transforms import transform_masks
from torch_mesmer.augmentation import MultiTransform
from torch_mesmer.utils import histogram_normalization, percentile_threshold

# Targets, normalisation and augmentation of the published Mesmer training recipe.
IN_TRANSFORMS = ('inner-distance', 'pixelwise')
TRANSFORMS_KWARGS = {
    'pixelwise': {'dilation_radius': 1},
    'inner-distance': {'erosion_width': 1, 'alpha': 'auto', 'beta': 1},
}
CLAHE_KERNEL_SIZE = 128   # offline multiplex_preprocess on the full image
ZOOM_MIN = 0.7            # zoom range (0.7, 1/0.7)


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


class SegmentationDataset(Dataset):

    def __init__(self, 
                 X, 
                 y,
                 dataset_type='train',
                 in_transforms=IN_TRANSFORMS,
                 transforms_kwargs=None,
                 crop_size = 256,
                 rotation_range=180,
                 zoom_min=ZOOM_MIN,
                 kernel_size=CLAHE_KERNEL_SIZE,
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

        y_semantic = []

        for compartment in range(y.shape[0]):
            for transform in self.in_transforms:
                
                y_compartment = np.expand_dims(y[compartment], 0)

                transform_kwargs = self.transforms_kwargs.get(transform, dict())
                y_transform = transform_masks(y_compartment, transform, unbatched=True,
                                                **transform_kwargs)
                y_semantic.append(y_transform.squeeze(axis=0))

        # Output will be of shape [C1T1, C1T2, C2T1, C2T2]
        y_semantic = np.concatenate(y_semantic, axis=0)

        # Output is now sum([C1T1, C1T2, C2T1, C2T2])

        return y_semantic

    def __len__(self):
            return self.X.shape[0]

    def __getitem__(self, idx):

        # Indexing for histogram normalization allows for no batches
        x = self.X[idx]
        y = self.y[idx]

        x = self._normalize(x)

        y = self._transform_labels(y)

        # Convert to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # Stack x and y along the channel dimension, so one draw of the
        # augmentation moves them together.
        combined = torch.cat([x, y], dim=0)

        transform = MultiTransform(
            combined.shape[1:],
            dataset_type=self.dataset_type,
            crop_size=self.crop_size,
            rotation_range=self.rotation_range,
            zoom_min=self.zoom_min,
        )
        combined_out = transform(combined, self.interpolation)

        # Split back into x and y
        x_out = combined_out[:x.shape[0]]
        y_out = combined_out[x.shape[0]:]

        return (x_out, y_out)
    
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
    semantic_heads=None,
    seed=0,
):

    # dataset will be fed in channels first, no exceptions!

    # Both loaders drop the last partial batch, as the published recipe does
    # (it runs n // batch_size steps per epoch). The train generator fixes the
    # shuffle order and, through it, each worker's augmentation draws (PyTorch
    # seeds torch, random and numpy in every worker from it).

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
