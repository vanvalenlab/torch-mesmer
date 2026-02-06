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
                 in_transforms=["inner-distance", "pixelwise"], 
                 augment=True, 
                 data_format='channels_first',
                 crop_size = 256,
                 rotation_range=180,
                 zoom=0.75,
                 preprocess=False,
                 target_mpp = 0.65,
                 semantic_heads = [1,3,1,3],
                 nuc_first = False,
                 ):
        
        self.X = X
        self.y = y
        self.in_transforms = in_transforms
        self.augment = augment

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

        # in the dataset, the first channel is the nucleus, but the first mask is whole cell
        # swap them when generating the dataset
        self.nuc_first = nuc_first

        # Convert to channels first format if necessary
        if self.data_format == 'channels_last':
            self.X = np.moveaxis(self.X, -1, 1)
            self.y = np.moveaxis(self.y, -1, 1)
            self.data_format = 'channels_first'

    def _normalize(self, X):

        X = np.expand_dims(X, 0)

        assert len(X.shape) == 4, 'add batch dimension'

        X_norm = histogram_normalization(X, data_format=self.data_format)
        X_norm = X_norm.squeeze(axis=0)

        return X_norm
    
    def _gen_augmentation(self):

        self.angle = random.uniform(-self.rotation_range, self.rotation_range)
        self.scale = int(random.uniform(self.zoom * self.crop_size, (1/self.zoom) * self.crop_size))
        self.do_hflip = random.random() > 0.5
        self.do_vflip = random.random() > 0.5

        self.bbox = transforms.RandomCrop.get_params(
            torch.rand(512,512), output_size=(self.scale, self.scale))

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
    
    def _augment(self, image, interpolation_mode = 'bilinear'):

        if interpolation_mode == 'bilinear':
            interpolation=F.InterpolationMode.BILINEAR

        elif interpolation_mode == 'nearest':
            interpolation = F.InterpolationMode.NEAREST
            
        # Random crop parameters
        i, j, h, w = self.bbox
        
        ### Apply to image (bilinear interpolation)
        image = F.rotate(image, self.angle, interpolation=interpolation, fill=-1)

        image = F.crop(image, i, j, h, w)

        image = F.resize(image, (self.crop_size, self.crop_size), 
                        interpolation=interpolation)
        
        if self.do_hflip:
            image = F.hflip(image)

        if self.do_vflip:
            image = F.vflip(image)

        return image

    def __len__(self):
            return self.X.shape[0]

    def __getitem__(self, idx):

        # Indexing for histogram normalization allows for no batches
        x = self.X[idx]
        y = self.y[idx]

        # nuc and cyto channels are swapped between X and y
        # Make sure they're swapped back so the output is in the same order as the input
        if self.nuc_first:
            y = np.flip(y, axis=0)

        x = self._normalize(x)

        y = self._transform_labels(y)

        # Convert to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if self.augment:

            # Init random augmentation parameters for this sample
            self._gen_augmentation()

            x = self._augment(x, interpolation_mode='bilinear')
            y = self._augment(y, interpolation_mode='bilinear')

        return (x, y)
    
def create_data_loaders(
    train,
    val,
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

    in_transforms = ["inner-distance", "pixelwise"]

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
            crop_size=crop_size,
            zoom=zoom_min,
            data_format=data_format,
            in_transforms=in_transforms, 
            augment=True)
        
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    if val is not None:
        val_dataset = SegmentationDataset(
            val['X'], 
            val['y'],
            crop_size=crop_size,
            zoom=zoom_min,
            data_format=data_format,
            in_transforms=in_transforms, 
            augment=False)  
      
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

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
