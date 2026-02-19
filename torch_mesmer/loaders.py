import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import zarr
from torch_mesmer.transforms import transform_masks
from torch_mesmer.augmentation import MultiTransform
from torch_mesmer.utils import histogram_normalization, percentile_threshold

class SegmentationDataset(Dataset):

    def __init__(self, 
                 X, 
                 y,
                 mpps,
                 dataset_type='train',
                 in_transforms=["inner-distance", "pixelwise"], 
                 augment=True, 
                 data_format='channels_first',
                 crop_size = 256,
                 rotation_range=180,
                 zoom=0.75,
                 preprocess=False,
                 target_mpp = 0.5,
                 semantic_heads = [1,3,1,3],
                 ):
    
        self.mpps = mpps
        self.X = X
        self.y = y

        if self.mpps is not None:

            good_mpps = np.argwhere(~np.isnan(self.mpps)).squeeze()

            self.X = self.X[good_mpps]
            self.y = self.y[good_mpps]
            self.mpps = self.mpps[good_mpps]
        
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

        self.data_format = data_format
        self.channel_axis = -1
        self.crop_size = crop_size
        self.rotation_range = rotation_range
        self.zoom = zoom
        self.preprocess = preprocess
        self.target_mpp = target_mpp
        self.semantic_heads = semantic_heads
        self.total_channels = sum(self.semantic_heads)

        # Convert to channels first format if necessary
        if self.data_format == 'channels_last':
            self.X = np.moveaxis(self.X, -1, 1)
            self.y = np.moveaxis(self.y, -1, 1)
            self.data_format = 'channels_first'

    def _normalize(self, X):

        X = np.expand_dims(X, 0)

        assert len(X.shape) == 4, 'add batch dimension'
        X_norm = percentile_threshold(X, percentile=99.9)
        X_norm = histogram_normalization(X_norm, data_format=self.data_format)
        X_norm = X_norm.squeeze(axis=0)

        return X_norm

    def _transform_labels(self, y):

        y_semantic = []

        for compartment in range(y.shape[0]):
            for transform in self.in_transforms:
                
                y_compartment = np.expand_dims(y[compartment], 0)

                transform_kwargs = self.transforms_kwargs.get(transform, dict())
                y_transform = transform_masks(y_compartment, transform, data_format=self.data_format, unbatched=True,
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
        mpp = self.mpps[idx]

        x = self._normalize(x)

        y = self._transform_labels(y)

        # Convert to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        combined = torch.cat([x, y], dim=0)
        combined_out = torch.zeros((combined.shape[0], self.crop_size, self.crop_size))
        
        transform = MultiTransform(mpp=mpp, target_mpp=self.target_mpp, dataset_type=self.dataset_type)

        # Stack x and y along channel dimension

        for c in range(combined.shape[0]):
            combined_out[c] = transform(combined[c], interpolation_mode=self.transform_type[c])
        
        # Split back into x and y
        x_out = combined_out[:x.shape[0]]
        y_out = combined_out[x.shape[0]:]
        
        return (x_out, y_out)
    
def create_data_loaders(
    train,
    val,
    crop_size=256,
    zoom_min=0.75,
    batch_size=16,
    num_workers=4,
    data_format = 'channels_first',
    in_transforms = ["inner-distance", "pixelwise"],
    semantic_heads = [1,3]
):

    
    dataloader = None
    valloader = None
    
    if train is not None:

        train_dataset = SegmentationDataset(
            train['X'], 
            train['y'],
            train['mpp'],
            crop_size=crop_size,
            dataset_type='train',
            zoom=zoom_min,
            data_format=data_format,
            in_transforms=in_transforms, 
            augment=True,
            semantic_heads=semantic_heads)
        
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    if val is not None:
        val_dataset = SegmentationDataset(
            val['X'], 
            val['y'],
            val['mpp'],
            crop_size=crop_size,
            dataset_type='val',
            zoom=zoom_min,
            data_format=data_format,
            in_transforms=in_transforms,
            augment=True,
            semantic_heads=semantic_heads)  
      
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return dataloader, valloader

if __name__ == '__main__':

    z_train = zarr.open("/data/shared/tissuenet/tissuenet_v1.1_train.zarr")
    z_val = zarr.open("/data/shared/tissuenet/tissuenet_v1.1_val.zarr")

    # Set up data generators with updated data
    train_data, val_data = create_data_loaders(
        z_train,
        z_val,
        crop_size=256,
        zoom_min=0.75,
        batch_size=1,
        data_format='channels_first',
        num_workers=4
    )

    train_iter = iter(train_data)
    sample = next(train_iter)
    print(sample[0].shape, sample[1].shape)


    # train_iter = iter(val_data)
    # sample = next(train_iter)
    # print(sample[0].shape, sample[1].shape)
