import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_mesmer.transforms import transform_masks
from torch_mesmer.augmentation import MultiTransform
from torch_mesmer.utils import histogram_normalization, percentile_threshold

# Zoom range of the published Mesmer training recipe: (0.7, 1/0.7) on each axis.
ZOOM_MIN = 0.7

class SegmentationDataset(Dataset):

    def __init__(self, 
                 X, 
                 y,
                 dataset_type='train',
                 in_transforms=["inner-distance", "pixelwise"], 
                 augment=True, 
                 crop_size = 256,
                 rotation_range=180,
                 zoom_min=ZOOM_MIN,
                 preprocess=False,
                 semantic_heads = [1,3,1,3],
                 ):
    
        self.X = X
        self.y = y

        self.in_transforms = in_transforms
        self.augment = augment
        self.dataset_type = dataset_type
        self.transform_type = [
            'bilinear',
            'bilinear',
            'bilinear',
            'nearest',
            'nearest',
            'nearest',
            'bilinear',
            'nearest',
            'nearest',
            'nearest'
        ]


        self.transforms_kwargs = {
            "pixelwise": {"dilation_radius": 1},
            "inner-distance": {"erosion_width": 1, "alpha": "auto"},
        }

        self.crop_size = crop_size
        self.rotation_range = rotation_range
        self.zoom_min = zoom_min
        self.preprocess = preprocess
        self.semantic_heads = semantic_heads
        self.total_channels = sum(self.semantic_heads)

    def _normalize(self, X):

        X = np.expand_dims(X, 0)

        assert len(X.shape) == 4, 'add batch dimension'
        X_norm = percentile_threshold(X, percentile=99.9)
        X_norm = histogram_normalization(X_norm)

        # Return without batch dimension for dataloader
        X_norm = X_norm.squeeze(axis=0)

        return X_norm

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
        combined_out = transform(combined, self.transform_type)

        # Split back into x and y
        x_out = combined_out[:x.shape[0]]
        y_out = combined_out[x.shape[0]:]

        # Undo background padding issues -- resets background padding as 1
        y_out[3] = (y_out[2]+y_out[1]) < 1
        y_out[7] = (y_out[6]+y_out[5]) < 1

        return (x_out, y_out)
    
def create_data_loaders(
    train,
    val,
    crop_size=256,
    zoom_min=ZOOM_MIN,
    rotation_range=180,
    batch_size=16,
    num_workers=4,
    in_transforms = ["inner-distance", "pixelwise"],
    semantic_heads = [1,3,1,3],
    seed=0,
):

    # dataset will be fed in channels first, no exceptions!

    # Both loaders drop the last partial batch, as the published recipe does
    # (it runs n // batch_size steps per epoch). The train generator fixes the
    # shuffle order and, through it, each worker's augmentation draws (PyTorch
    # seeds torch, random and numpy in every worker from it).

    dataloader = None
    valloader = None
    
    if train is not None:

        train_dataset = SegmentationDataset(
            train['X'], 
            train['y'],
            crop_size=crop_size,
            dataset_type='train',
            rotation_range=rotation_range,
            zoom_min=zoom_min,
            in_transforms=in_transforms, 
            augment=True,
            semantic_heads=semantic_heads)
        
        generator = torch.Generator()
        generator.manual_seed(seed)

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=True, drop_last=True, generator=generator)

    if val is not None:
        val_dataset = SegmentationDataset(
            val['X'], 
            val['y'],
            crop_size=crop_size,
            dataset_type='val',
            rotation_range=rotation_range,
            zoom_min=zoom_min,
            in_transforms=in_transforms,
            augment=True,
            semantic_heads=semantic_heads)  
      
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                               pin_memory=True, drop_last=True)

    return dataloader, valloader
