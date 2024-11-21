from torch.utils.data import Dataset, DataLoader
from deepcell.image_generators import _transform_masks
import numpy as np
import torch

class SemanticDataset(Dataset):
    def __init__(self, X, y, transforms=['outer-distance'], transforms_kwargs={}):
        self.X = X
        self.y = y
        self.transforms = transforms
        self.transforms_kwargs = transforms_kwargs
        self.channel_axis=1

    def _transform_labels(self, y):
        y_semantic_list = []
        # loop over channels axis of labels in case there are multiple label types
        for label_num in range(y.shape[self.channel_axis]):
    
            if self.channel_axis == 1:
                y_current = y[:, label_num:label_num + 1, ...]
            else:
                y_current = y[..., label_num:label_num + 1]
    
            data_format='channels_first'
            for transform in self.transforms:
                transform_kwargs = self.transforms_kwargs.get(transform, dict())
                y_transform = _transform_masks(y_current, transform,
                                               data_format=data_format,
                                               **transform_kwargs)
                y_semantic_list.append(y_transform)

        y_semantic_list = [ys[0] for ys in y_semantic_list]
        return y_semantic_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]
        y_semantic_list = self._transform_labels(self.y[idx:idx+1]) 
        return (x, y_semantic_list)


class CroppingDatasetTorch(Dataset):
    def __init__(self, X, y, rotation_range, shear_range, zoom_range, horizontal_flip, vertical_flip, crop_size, batch_size=8, transforms=['outer-distance'], transforms_kwargs={}):
        self.X = X
        self.y = y
        self.transforms = transforms
        self.transforms_kwargs = transforms_kwargs
        self.channel_axis=3
        self.seed = 0
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.crop_size = (crop_size, crop_size)
        self.batch_size = batch_size
        
    def _transform_labels(self, y):
        y_semantic_list = []
        # loop over channels axis of labels in case there are multiple label types
        for label_num in range(y.shape[self.channel_axis]):
    
            if self.channel_axis == 1:
                y_current = y[:, label_num:label_num + 1, ...]
            else:
                y_current = y[..., label_num:label_num + 1]

            data_format='channels_last'
            for transform in self.transforms:
                transform_kwargs = self.transforms_kwargs.get(transform, dict())
                y_transform = _transform_masks(y_current, transform,
                                               data_format=data_format,
                                               **transform_kwargs)
                y_semantic_list.append(y_transform)

        y_semantic_list = [ys[0] for ys in y_semantic_list]
        return y_semantic_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.seed is not None and idx%self.batch_size==0:
            np.random.seed(self.seed + idx//self.batch_size)
            torch.manual_seed(self.seed+idx//self.batch_size)

        from torchvision.transforms import v2 as transforms

        # Create a Compose object with a list of transformations
        transform = transforms.Compose([
            transforms.ToImage(),          # Convert the image to a PyTorch tensor
            # transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomCrop(self.crop_size),
            transforms.RandomRotation(degrees=self.rotation_range),
            transforms.RandomResizedCrop(size=256, scale=self.zoom_range),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

        x = self.X[idx]
        y_semantic_list = self._transform_labels(self.y[idx:idx+1]) 

        # state = torch.get_rng_state()

        # x = transform(x)
        # for k in range(len(y_semantic_list)):
        #     torch.set_rng_state(state)
        #     y_semantic_list[k] = transform(y_semantic_list[k])

        li = transform([x]+[y_semantic_list[k] for k in range(len(y_semantic_list))])
        x = li[0]
        y_semantic_list = li[1:]
        
        return (x, y_semantic_list)