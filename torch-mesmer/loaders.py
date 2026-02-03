import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
import zarr
from transforms import transform_masks
from utils import histogram_normalization
from torchvision.transforms.v2 import functional as F
import random

class SegmentationDataset(Dataset):

    def __init__(self, 
                 X, 
                 y,
                 mpps = None, 
                 in_transforms=['outer-distance'], 
                 transforms_kwargs={}, 
                 augment=True, 
                 data_format='channels_first',
                 crop_size = 256,
                 rotation_range=180,
                 zoom=0.75,
                 preprocess=False,
                 target_mpp = 0.65
                 ):
        
        self.X = X[:]
        self.y = y[:]
        self.in_transforms = in_transforms
        self.augment = augment
        self.transforms_kwargs = transforms_kwargs
        self.data_format = data_format
        self.channel_axis = -1
        self.crop_size = crop_size
        self.rotation_range = rotation_range
        self.zoom = zoom
        self.preprocess = preprocess
        self.mpps = mpps
        self.target_mpp = target_mpp

        # Filter out where pixel size is not defined
        # Need pixel size to be fixed for model
        if self.mpps is not None:

            good_mpps = ~np.isnan(self.mpps)

            self.X = self.X[good_mpps]
            self.y = self.y[good_mpps]
            self.mpps = self.mpps[good_mpps]

        # Convert to channels first format if necessary
        if self.data_format == 'channels_last':
            self.X = np.moveaxis(self.X, -1, 1)
            self.y = np.moveaxis(self.y, -1, 1)
            self.data_format = 'channels_first'

        if self.preprocess:
            print("Preprocessing...")
            self._preprocess()

    def _normalize(self, X):

        X = np.expand_dims(X, 0)

        assert len(X.shape) == 4, 'add batch dimension'

        X_norm = histogram_normalization(X, data_format=self.data_format)

        X_norm = X_norm.squeeze(axis=0)

        return X_norm

    def _preprocess(self):

        print("Histogram normalizing...")
        self.X_norm = histogram_normalization(self.X, data_format=self.data_format)
        print("Normalization done.")

        print("Transforming labels...")

        # Splitting labels for augmentation
        self.semantic_continuous, self.semantic_discrete = self._transform_labels(self.y)


    def _transform_labels(self, y):

        y_semantic_list = []

        for transform in self.in_transforms:
            transform_kwargs = self.transforms_kwargs.get(transform, dict())

            y_transform = transform_masks(y, transform, data_format=self.data_format,
                                            **transform_kwargs)
            
            y_semantic_list.append(y_transform)

        # split at axis 1 (B, 2, H, W) based on discrete and continuous functions

        semantic_continuous = np.concat(y_semantic_list[:2], axis=1)
        semantic_discrete = np.concat(y_semantic_list[2:], axis=1)

        return semantic_continuous, semantic_discrete
    
    def _augment(self, image, mask_continuous, mask_discrete, scale_factor):

        # Generate random parameters once
        angle = random.uniform(-self.rotation_range, self.rotation_range)

        # Zoom parameters
        scale = (
            int(random.uniform(self.zoom * self.crop_size, (1/self.zoom) * self.crop_size) * scale_factor),
            int(random.uniform(self.zoom * self.crop_size, (1/self.zoom) * self.crop_size) * scale_factor)
        )

        # Random crop parameters
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=scale)
        
        # Flip parameters
        do_hflip = random.random() > 0.5
        do_vflip = random.random() > 0.5
        
        ### Apply to image (bilinear interpolation)
        image = F.crop(image, i, j, h, w)
        image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)
        image = F.resize(image, (self.crop_size, self.crop_size), 
                        interpolation=F.InterpolationMode.BILINEAR)
        if do_hflip:
            image = F.hflip(image)
        if do_vflip:
            image = F.vflip(image)

        
        ### Apply to continuous masks (bilinear interpolation)

        mask_continuous = F.crop(mask_continuous, i, j, h, w)
        mask_continuous = F.rotate(mask_continuous, angle, 
                                    interpolation=F.InterpolationMode.BILINEAR)
        mask_continuous = F.resize(mask_continuous, (self.crop_size, self.crop_size),
                                    interpolation=F.InterpolationMode.BILINEAR)
        if do_hflip:
            mask_continuous = F.hflip(mask_continuous)
        if do_vflip:
            mask_continuous = F.vflip(mask_continuous)
        

        ### Apply to discrete masks (nearest-neighbor interpolation)

        mask_discrete = F.crop(mask_discrete, i, j, h, w)
        mask_discrete = F.rotate(mask_discrete, angle, 
                                interpolation=F.InterpolationMode.NEAREST)
        mask_discrete = F.resize(mask_discrete, (self.crop_size, self.crop_size),
                                interpolation=F.InterpolationMode.NEAREST)
        if do_hflip:
            mask_discrete = F.hflip(mask_discrete)
        if do_vflip:
            mask_discrete = F.vflip(mask_discrete)

        mask_discrete = torch.cat([
            torch.logical_not(mask_discrete.clone()),
            mask_discrete
        ], axis=0)
        
        return image, mask_continuous, mask_discrete

    def __len__(self):
            return self.X.shape[0]

    def __getitem__(self, idx):

        scale_factor = self.target_mpp / self.mpps[idx]

        # Indexing for histogram normalization allows for no batches
        x = self._normalize(self.X[idx])

        # Indexing for transformations requires batches -- use slicing
        semantic_continuous, semantic_discrete = self._transform_labels(self.y[slice(idx, idx+1)])

        # Convert to tensors
        x = torch.from_numpy(x).float()
        semantic_continuous = torch.from_numpy(semantic_continuous).float()
        semantic_discrete = torch.from_numpy(semantic_discrete).float()

        if self.augment:
            # crop and also augment
            x, semantic_continuous, semantic_discrete = self._augment(x, semantic_continuous, semantic_discrete, scale_factor)
            
            y = torch.cat([
                semantic_continuous.squeeze(),
                semantic_discrete.squeeze()
            ], axis=0)
        
        else:
            scale = (
                int(self.crop_size * scale_factor),
                int(self.crop_size * scale_factor)
            )

            i, j, h, w = transforms.RandomCrop.get_params(
                x, output_size=scale)
                        
            semantic_discrete = torch.cat([
                torch.logical_not(semantic_discrete.clone()),
                semantic_discrete
            ], axis=0)

            y = torch.cat([
                semantic_continuous.squeeze(),
                semantic_discrete.squeeze()
            ], axis = 0)

            x = F.crop(x, i, j, h, w)
            y = F.crop(y, i, j, h, w)

        return (x, y)
    
def create_data_loaders(
    train,
    val,
    train_mpps = None,
    val_mpps = None,
    crop_size=256,
    zoom_min=0.75,
    batch_size=16,
    outer_erosion_width=1,
    inner_distance_alpha="auto",
    inner_distance_beta=1,
    inner_erosion_width=0,
    num_workers=4,
    data_format = 'channels_first',
    preprocess = True
):

    in_transforms = ["inner-distance", "outer-distance", "fgbg"]

    transforms_kwargs = {

        "outer-distance": {
            "erosion_width": outer_erosion_width
            },

        "inner-distance": {
            "alpha": inner_distance_alpha,
            "beta": inner_distance_beta,
            "erosion_width": inner_erosion_width,
            },

    }
    
    dataloader = None
    valloader = None
    
    if train is not None:

        train_dataset = SegmentationDataset(
            train['X'], 
            train['y'],
            mpps = train_mpps,
            crop_size=crop_size,
            zoom=zoom_min,
            data_format=data_format,
            in_transforms=in_transforms, 
            augment=True,
            preprocess=preprocess,
            transforms_kwargs=transforms_kwargs)
        
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    if val is not None:
        val_dataset = SegmentationDataset(
            val['X'], 
            val['y'],
            mpps = val_mpps, 
            crop_size=crop_size,
            zoom=zoom_min,
            data_format=data_format,
            in_transforms=in_transforms, 
            augment=True,
            preprocess=preprocess,
            transforms_kwargs=transforms_kwargs)  
      
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return dataloader, valloader

if __name__ == '__main__':

    config = {
        'model_path': "data/segmentation/model/",
        'data_path': 'data/DynamicNuclearNet-segmentation-v1_0',
        'run_info': 'data/segmentation/logs/',
        'epochs': 16,
        'seed': 0,
        'min_objects': 1,
        'zoom_min': 0.75,
        'batch_size': 12,
        'backbone': 'efficientnetv2bl',
        'crop_size': 256,
        'lr': 1e-4,
        'outer_erosion_width': 1,
        'inner_distance_alpha': 'auto',
        'inner_distance_beta': 1,
        'inner_erosion_width': 0,
        'pyramid_levels': "P3-P4-P5-P6-P7",
        'num_workers': 16
    }

    z_train = zarr.open(f"{config['data_path']}/train.zarr")
    z_val = zarr.open(f"{config['data_path']}/val.zarr")

    # Set up data generators with updated data
    train_data, val_data = create_data_loaders(
        z_train,
        z_val,
        crop_size=config['crop_size'],
        zoom_min=config['zoom_min'],
        batch_size=config['batch_size'],
        outer_erosion_width=config['outer_erosion_width'],
        inner_distance_alpha=config['inner_distance_alpha'],
        inner_distance_beta=config['inner_distance_beta'],
        inner_erosion_width=config['inner_erosion_width'],
        preprocess=False,
        data_format='channels_last'
    )

    train_iter = iter(train_data)
    sample = next(train_iter)
    print(sample[0].shape, sample[1].shape)
