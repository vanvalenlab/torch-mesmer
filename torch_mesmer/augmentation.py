import torch
from torch import nn
from torchvision.transforms import v2 as transforms
from torchvision.transforms import functional as F
import random

class MultiTransform(nn.Module):

    def __init__(self, mpp, target_mpp = 0.5, dataset_type = 'val', rotation_range=180, crop_size=256, zoom=0.75):
        super().__init__()

        '''
        Get context:
        - MPP
        - validation/training

        Make random tranformations:
        - Rotation degree
        - New shape from old shape (resize)
        - Random crop
        - Horizontal flip p-value
        - vertical flip p-value

        Forward transforms:
        - MPP
        - Binary or continuous

        '''
        self.target_mpp = target_mpp
        self.mpp = mpp
        self.dataset_type = dataset_type
        self.crop_size = crop_size
        self.rotation_range = rotation_range
        self.zoom = zoom
        self._gen_augmentation_params()


    def _gen_augmentation_params(self):

        resize_ratio = self.mpp/self.target_mpp
        self.scale = int(self.crop_size * resize_ratio)

        if self.dataset_type == 'train':
            self.angle = random.uniform(-self.rotation_range, self.rotation_range)
            self.zoom_range = int(random.uniform(self.zoom * self.crop_size, (1/self.zoom) * self.crop_size))
            self.do_hflip = random.random() > 0.5
            self.do_vflip = random.random() > 0.5
            self.bbox = transforms.RandomCrop.get_params(
                torch.rand(512,512), output_size=(self.zoom_range, self.zoom_range))

    def forward(self, image, interpolation_mode = 'bilinear'):

        image = image.unsqueeze(0)

        if interpolation_mode == 'bilinear':
            interpolation=F.InterpolationMode.BILINEAR

        elif interpolation_mode == 'nearest':
            interpolation = F.InterpolationMode.NEAREST
            
        if self.dataset_type == 'train':
            i, j, h, w = self.bbox
            image = F.crop(image, i, j, h, w)

        # Resize based on MPP
        image = F.resize(image, (self.scale, self.scale), 
                        interpolation=interpolation)
        
        ### Apply augmentations to image if necessary (training)
        if self.dataset_type == 'train':

            image = F.rotate(image, self.angle, interpolation=interpolation, fill=0)

            if self.do_hflip:
                image = F.hflip(image)

            if self.do_vflip:
                image = F.vflip(image)

        # Combined crop to model size and pad if necessary
        image = F.crop(image, 0, 0, self.crop_size, self.crop_size)
        image = image.squeeze(0)

        return image