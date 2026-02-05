import torch
from torch.nn import functional as F
from torchvision.transforms import functional as fvision

import numpy as np
from tqdm import tqdm

from utils import resize, histogram_normalization
from model import PanopticNet
import math
import skimage
from postprocess_utils import merge_nearby_points
from skimage.measure import regionprops


class Mesmer():

    def __init__(
            self, 
            model_path=None, 
            device=None, 
            postprocess_kwargs=None,
            batch_size = 16,
            data_format = 'channels_first'
    ):

        if device is None:
            self.device = 'cpu'
        else:
            self.device=device

        if model_path is None:
            raise Exception("Please provide a path to the model checkpoint file.")
        
        print("Initializing model...")
        model = PanopticNet(
            crop_size=256,
            backbone='resnet50',
            pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
            backbone_levels=['C3', 'C4', 'C5'],
            n_semantic_classes=[1,3,1,3]
        ).to(self.device)

        # Dummy data to make semantic heads
        dummy = torch.rand(1, 2, model.crop_size, model.crop_size).to(self.device)
        _ = model(dummy)
        del dummy

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)

        print(f"Model initialized. \n   Using device: {device}")
        print()

        self.device = device
        self.model = model.eval()
        self.postprocess_kwargs=postprocess_kwargs
        self.batch_size = batch_size
        self.data_format = data_format

        self.image_shape = model.crop_size
        self.in_channels = 1
        self.out_channels = 4
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.model_mpp = 0.65
            
        self.n_iter = self.postprocess_kwargs.get('n_iter', 200)
        self.step_size = self.postprocess_kwargs.get('step_size', 0.1)
        self.postprocess_method = self.postprocess_kwargs.get('postprocess_method','classical')
        self.transform_thresh = self.postprocess_kwargs.get('transform_thresh', 0.05)
        self.reduced_thresh = self.postprocess_kwargs.get('reduced_thresh', 0.05)
        self.relevant_counts = self.postprocess_kwargs.get('relevant_counts', 20)

        if self.postprocess_kwargs['small_objects_threshold'] == 'auto':
            self.small_objects_threshold = np.pi * (self.postprocess_kwargs['radius']/2) ** 2
        else:
            self.small_objects_threshold = self.postprocess_kwargs['small_objects_threshold']

    def _preprocess(self, x):

        assert len(x.shape) == 4, 'add batch dimension'

        x = histogram_normalization(x, data_format=self.data_format)

        return x

    def _resize_input(self, x):

        upscale_H = self.H * self.resize_factor
        upscale_W = self.W * self.resize_factor
        new_size = (int(upscale_H), int(upscale_W))

        x = fvision.resize(x, new_size, interpolation = fvision.InterpolationMode.BILINEAR)

        return x
    
    def _unfold_with_overlap(self, x, overlap=32):
        """
        Extract overlapping tiles from image sequence
        image: (T, C, H, W) tensor
        Returns tiles of shape (T*n_tiles_h*n_tiles_w, C, tile_size, tile_size)
        """
        self.curr_batch_size, C, H, W = x.shape

        stride = self.image_shape - overlap
        
        # Calculate number of tiles needed - ensure we cover the entire image
        nh = int(np.ceil((H - self.image_shape) / stride)) + 1
        nw = int(np.ceil((W - self.image_shape) / stride)) + 1
        
        # Process each frame independently, then stack
        all_tiles = []
        for t in range(self.curr_batch_size):
            frame = x[t]  # (C, H, W)
            frame_tiles = []
            
            for i in range(nh):
                for j in range(nw):
                    # For interior tiles, use regular stride
                    # For the last tile, align to the right/bottom edge
                    if i == nh - 1:
                        h_start = H - self.image_shape
                    else:
                        h_start = i * stride
                        
                    if j == nw - 1:
                        w_start = W - self.image_shape
                    else:
                        w_start = j * stride
                    
                    tile = frame[:, h_start:h_start + self.image_shape, w_start:w_start + self.image_shape]
                    frame_tiles.append(tile)
            
            frame_tiles = torch.stack(frame_tiles, dim=0)  # (nh*nw, C, tile_size, tile_size)
            all_tiles.append(frame_tiles)
        
        # Stack all frames
        all_tiles = torch.cat(all_tiles, dim=0)  # (T*nh*nw, C, tile_size, tile_size)
        
        return all_tiles
    
    def _refold_with_blend(self, tiles, overlap=32):
        """
        Reconstruct image sequence from overlapping tiles using weighted blending
        tiles: (T*n_tiles_h*n_tiles_w, C, tile_size, tile_size)
        original_shape: (T, C, H, W)
        """
        T = self.curr_batch_size
        C = self.out_channels

        stride = self.image_shape - overlap

        upscale_H = int(self.H * self.resize_factor)
        upscale_W = int(self.W * self.resize_factor)
        
        # Calculate number of tiles per dimension (must match tile_with_overlap)
        nh = int(np.ceil((upscale_H - self.image_shape) / stride)) + 1
        nw = int(np.ceil((upscale_W - self.image_shape) / stride)) + 1
        tiles_per_frame = nh * nw
        
        # Create blending weight matrix
        weight_tile = self._create_blend_mask(self.image_shape, overlap, tiles.device)
        
        # Create output tensor for all frames
        output = torch.zeros(T, C, upscale_H, upscale_W, device=tiles.device)
        weights = torch.zeros(T, C, upscale_H, upscale_W, device=tiles.device)
        
        # Process each frame
        for t in range(T):
            # Get tiles for this frame
            frame_tiles = tiles[t * tiles_per_frame:(t + 1) * tiles_per_frame]
            
            # Reconstruct this frame
            idx = 0
            for i in range(nh):
                for j in range(nw):
                    # Match the tiling logic exactly
                    if i == nh - 1:
                        h_start = upscale_H - self.image_shape
                    else:
                        h_start = i * stride
                        
                    if j == nw - 1:
                        w_start = upscale_W - self.image_shape
                    else:
                        w_start = j * stride
                    
                    h_end = h_start + self.image_shape
                    w_end = w_start + self.image_shape
                    
                    # Apply the weight mask to this tile
                    output[t, :, h_start:h_end, w_start:w_end] += \
                        frame_tiles[idx] * weight_tile
                    weights[t, :, h_start:h_end, w_start:w_end] += weight_tile
                    idx += 1
        
        # Avoid division by zero
        weights = torch.clamp(weights, min=1e-8)
        
        return output / weights

    def _create_blend_mask(self, tile_size, overlap, device):
        """Create separable blending mask - computed once, reused for all tiles"""
        mask_1d = torch.ones(tile_size, device=device)
        fade = torch.linspace(0, 1, overlap, device=device)
        mask_1d[:overlap] = fade
        mask_1d[-overlap:] = fade.flip(0)
        
        # Create 2D mask via outer product
        mask = mask_1d.unsqueeze(1) * mask_1d.unsqueeze(0)
        return mask
    
    def _predict(self, x):

        x_predicted = []

        n_batch = x.shape[0]

        pbar = enumerate(tqdm(range(0, n_batch, self.batch_size), desc="Inference on batch", leave=False, colour='#FFDB58'))

        for _, i in pbar:

            batch = x[i:i+self.batch_size]

            with torch.inference_mode():
                pred = self.model(batch)
            
            x_predicted.append(pred)

        x_predicted = torch.cat(x_predicted, dim=0)

        return x_predicted
    
    def _resize_output(self, x):
        
        x = fvision.resize(x, (self.H, self.W), interpolation=fvision.InterpolationMode.BILINEAR)

        return x
    

    def _get_gradients(self, transform, foreground_tensor):
        # Move to device and ensure correct dtypes

        transform = torch.where(foreground_tensor > self.postprocess_kwargs['transform_thresh'], transform, -1).float()
        
        # Compute gradients of INNER distance (points toward peaks)
        transform_4d = transform.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=torch.float32, device=transform.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=torch.float32, device=transform.device).view(1, 1, 3, 3)
        
        
        gx = F.conv2d(transform_4d, sobel_x, padding=1)  # (1, 1, H, W)
        gy = F.conv2d(transform_4d, sobel_y, padding=1)  # (1, 1, H, W)

        # Normalize gradients
        grad_mag = torch.sqrt(gx**2 + gy**2) + 1e-8
        gx = gx / grad_mag
        gy = gy / grad_mag

        return gx, gy
    
    def _get_positions(self, transform):

        # Initialize positions for all pixels
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.H, device=transform.device, dtype=torch.float32),
            torch.arange(self.W, device=transform.device, dtype=torch.float32),
            indexing='ij'
        )

        positions = torch.stack([y_coords, x_coords], dim=-1)  # (H, W, 2) -> [y, x]

        return positions
    
    def _follow_flows(self, positions, gx, gy, niter=20, step_size=0.5):

        for step in range(niter):
            # Normalize positions to [-1, 1] for grid_sample
            norm_x = 2 * positions[..., 1] / (self.W - 1) - 1
            norm_y = 2 * positions[..., 0] / (self.H - 1) - 1
            grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, H, W, 2) in (x, y) order

            # Sample gradients at current positions
            gx_sample = F.grid_sample(
                gx, 
                grid, 
                align_corners=False, 
            ).squeeze()  # (H, W)

            gy_sample = F.grid_sample(
                gy, 
                grid, 
                align_corners=False, 
            ).squeeze()   # (H, W)

            # Update positions
            positions[..., 0] = positions[..., 0] + step_size * gy_sample
            positions[..., 1] = positions[..., 1] + step_size * gx_sample


            # Clamp to image bounds
            positions[..., 0].clamp_(0, self.H - 1)
            positions[..., 1].clamp_(0, self.W - 1)    

        return positions
    
    def _postprocess(self, x):

        label_image = np.zeros((self.curr_batch_size, 1, self.H, self.W), dtype=int)

        pbar = tqdm(range(self.curr_batch_size), desc="Postprocessing", leave=False, colour='#CE2029')

        for t in pbar:

            x_inner = x[t, 0].cpu().numpy()
            x_outer = x[t, 1].cpu().numpy()
            x_foreground = x[t, 3].cpu().numpy()
            x_background = x[t, 2].cpu().numpy()

            if self.postprocess_method == 'classical':

                markers = skimage.morphology.h_maxima(
                    x_inner, 
                    h=self.postprocess_kwargs['maxima_threshold'], 
                    footprint=skimage.morphology.disk(self.postprocess_kwargs['radius'])
                )

            if self.postprocess_method == 'hybrid':

                positions = self._get_positions(x[t, 0])
                gx, gy = self._get_gradients(x[t, 0], x[t, 3])
                positions = self._follow_flows(positions, gx, gy, niter=self.n_iter, step_size=self.step_size)

                inds = torch.argwhere(x[t,3] > self.transform_thresh).t().cpu().numpy()

                relevant = positions[inds[0], inds[1]].cpu().numpy().astype(int)
                relevant, relevant_counts = np.unique(relevant, axis=0, return_counts=True)

                relevant = relevant[relevant_counts > self.relevant_counts]
                relevant = merge_nearby_points(relevant, r=self.postprocess_kwargs['radius'])

                markers = np.zeros((self.H, self.W))

                for i in range(relevant.shape[0]):
                    curr_point = relevant[i]
                    markers[curr_point[0], curr_point[1]] = 1

            markers = skimage.measure.label(markers)

            x_inner = skimage.filters.gaussian(x_inner, sigma=1, channel_axis=0)
            x_outer = skimage.filters.gaussian(x_outer, sigma=1, channel_axis=0)

            label_temp = skimage.segmentation.watershed(
                -1 * (x_inner + x_outer), 
                markers, 
                mask= x_foreground > self.reduced_thresh, 
                watershed_line=True
            )

            for prop in regionprops(label_temp.squeeze()):
                label_ = prop.label
                if prop.eccentricity > self.postprocess_kwargs['eccentricity']:
                    label_temp = np.where(label_temp == label_, 0, label_temp)
                    continue
                if prop.area < 2.:
                    label_temp = np.where(label_temp == label_, 0, label_temp)
                    continue
                if prop.euler_number < 1:
                    bbox = prop.bbox
                    label_temp[bbox[0]:bbox[2], bbox[1]:bbox[3]] = prop.image_filled

            label_image[t] = skimage.morphology.area_closing(label_temp.squeeze())
            label_image[t], _, _ = skimage.segmentation.relabel_sequential(label_temp)

            
        label_image = label_image.astype(int)

        return label_image
        
    def segment(self,
                x,
                mpps = None,
                data_format='channels_first',
                return_transforms = False):

        if data_format == 'channels_last':
            x = np.moveaxis(x, -1, 1)

        label_image = np.zeros_like(x)

        # Keep track of original shape for rescaling after processing
        self.H = x.shape[-2]
        self.W = x.shape[-1]
        self.n_frames = x.shape[0]
        
        if mpps is not None:
            mpps = self.model_mpp / mpps
        else:
            mpps = np.ones((self.n_frames,))

        pbar = tqdm(range(0, self.n_frames, self.batch_size), leave=False, colour='#008080')

        transforms = torch.zeros((self.n_frames, self.out_channels, self.H, self.W))

        # Preprocess the images and resize to square if necessary
        for i in pbar:
            self.resize_factor = mpps[i]
            pbar.set_description(f"Histogram normalizing")
            x_batch = x[i:i+self.batch_size]
            
            x_batch = self._preprocess(x_batch)            

            x_batch = torch.from_numpy(x_batch).to(self.device)
            x_batch = self._resize_input(x_batch)

            pbar.set_description("Inference on tiles")

            # Unfold images for tiling
            tiles = self._unfold_with_overlap(x_batch, overlap=32)
            # Batchwise predictions
            
            tiles = tiles.float()
            output_tiles = self._predict(tiles)

            # Refold batch * tiles into shape (T, C, H_sq, W_sq)
            output_images = self._refold_with_blend(output_tiles, overlap=32)
            output_images = self._resize_output(output_images)

            # Reshape image back to original size
            pbar.set_description(f"Postprocessing using {self.postprocess_method}")

            if return_transforms:
                transforms[i:i+self.batch_size] = output_images
            
            label_image[i:i+self.batch_size] = self._postprocess(output_images)
                    
        pbar.set_description("Done segmenting images.")

        if return_transforms:
            return label_image, transforms
        else:
            return label_image

    

