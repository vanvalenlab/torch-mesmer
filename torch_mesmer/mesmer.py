import torch
from torch.nn import functional as F
from torchvision.transforms import functional as fvision

import numpy as np
from tqdm import tqdm

from .utils import resize, histogram_normalization
from .model import PanopticNet
import math
import skimage
from .postprocess_utils import merge_nearby_points
from skimage.measure import regionprops


class Mesmer():

    def __init__(
            self, 
            model_path=None, 
            device=None, 
            postprocess_kwargs={},
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
        self.in_channels = 2
        self.out_channels = 8
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.model_mpp = 0.65
            
        self.n_iter = self.postprocess_kwargs.get('n_iter', 200)
        self.step_size = self.postprocess_kwargs.get('step_size', 0.1)
        self.postprocess_method = self.postprocess_kwargs.get('postprocess_method','hybrid')
        self.transform_thresh = self.postprocess_kwargs.get('transform_thresh', 0.05)
        self.reduced_thresh = self.postprocess_kwargs.get('reduced_thresh', 0.05)
        self.maxima_threshold = self.postprocess_kwargs.get('maxima_threshold', 0.05)
        self.relevant_counts = self.postprocess_kwargs.get('relevant_counts', 10)
        self.small_objects_threshold = self.postprocess_kwargs.get('small_objects_threshold', 16)
        self.radius = self.postprocess_kwargs.get('radius', 10)
        self.eccentricity = self.postprocess_kwargs.get('radius', 0.9)

        self.n_compartment_predictions = self.out_channels / self.in_channels

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

        transform = torch.where(foreground_tensor > self.transform_thresh, transform, -1).float()
        
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
    
    def _get_positions(self, transform, downsample_factor=1):

        # Initialize positions for downsampled grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(0, self.H, downsample_factor, device=transform.device, dtype=torch.float32),
            torch.arange(0, self.W, downsample_factor, device=transform.device, dtype=torch.float32),
            indexing='ij'
        )

        positions = torch.stack([y_coords, x_coords], dim=-1)  # (H//ds, W//ds, 2) -> [y, x]

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
            x_peri = x[t, 1].cpu().numpy()
            x_foreground = x[t, 2].cpu().numpy()
            # x_background = x[t, 3].cpu().numpy()

            if self.postprocess_method == 'classical':

                markers = skimage.morphology.h_maxima(
                    x_inner, 
                    h=self.maxima_threshold, 
                    footprint=skimage.morphology.disk(self.radius)
                )

            if self.postprocess_method == 'hybrid':

                # Downsample factor - increase for larger images to save memory
                # For a 2048x2048 image, downsample_factor=4 gives you a 512x512 grid
                downsample_factor = max(1, min(self.H, self.W) // 2048)
                print("following flows")
                positions = self._get_positions(x[t, 0], downsample_factor=downsample_factor)
                gx, gy = self._get_gradients(x[t, 0], x[t, 2])
                positions = self._follow_flows(positions, gx, gy, niter=self.n_iter, step_size=self.step_size)
                print('flows followed')
                # Sample the background/foreground at downsampled positions
                # Normalize positions to [-1, 1] for grid_sample
                norm_x = 2 * positions[..., 1] / (self.W - 1) - 1
                norm_y = 2 * positions[..., 0] / (self.H - 1) - 1
                grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, H_ds, W_ds, 2)
                
                background_sampled = F.grid_sample(
                    x[t, 3].unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                    grid,
                    align_corners=False,
                ).squeeze()  # (H_ds, W_ds)
                
                # Get indices where background is low (foreground pixels)
                inds = torch.argwhere(background_sampled < self.transform_thresh).cpu().numpy()
                
                # Get the converged positions for these foreground points
                relevant = positions[inds[:, 0], inds[:, 1]].cpu().numpy().astype(int)
                relevant, relevant_counts = np.unique(relevant, axis=0, return_counts=True)

                relevant = relevant[relevant_counts > self.relevant_counts]
                relevant = merge_nearby_points(relevant, r=self.radius)

                markers = np.zeros((self.H, self.W))

                for i in range(relevant.shape[0]):
                    curr_point = relevant[i]
                    markers[curr_point[0], curr_point[1]] = 1

            markers = skimage.measure.label(markers)

            x_inner = skimage.filters.gaussian(x_inner, sigma=1, channel_axis=0)
            print('watershed')
            label_temp = skimage.segmentation.watershed(
                -1 * x_inner, 
                markers, 
                mask= x_foreground + x_peri > self.reduced_thresh, 
                watershed_line=False
            )
            print('watershed done')

            # for prop in regionprops(label_temp.squeeze()):
            #     label_ = prop.label
            #     if prop.eccentricity > self.eccentricity:
            #         label_temp = np.where(label_temp == label_, 0, label_temp)
            #         continue
            #     if prop.area < 2.:
            #         label_temp = np.where(label_temp == label_, 0, label_temp)
            #         continue
            #     if prop.euler_number < 1:
            #         bbox = prop.bbox
            #         label_temp[bbox[0]:bbox[2], bbox[1]:bbox[3]] = prop.image_filled

            # label_image[t] = skimage.morphology.area_closing(label_temp.squeeze(), area_threshold=self.small_objects_threshold)
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

        # Assume single image input - add batch dimension if needed
        if len(x.shape) == 3:
            x = x[np.newaxis, ...]
        
        print(f"Input shape: {x.shape}")

        # Keep track of original shape
        self.H = x.shape[-2]
        self.W = x.shape[-1]
        self.curr_batch_size = 1  # Processing single image
        
        if mpps is not None:
            self.resize_factor = self.model_mpp / mpps
        else:
            self.resize_factor = 1.0

        # Preprocess the image (on CPU)
        print("Preprocessing image...")
        x = self._preprocess(x)
        
        # Convert to tensor but keep on CPU for now
        x_tensor = torch.from_numpy(x)
        
        # Resize on CPU
        print("Resizing image...")
        x_resized = self._resize_input(x_tensor)
        
        # Unfold into tiles (on CPU) - this doesn't load to GPU yet
        print("Tiling image...")
        tiles = self._unfold_with_overlap(x_resized, overlap=32)
        tiles = tiles.float()
        
        n_tiles = tiles.shape[0]
        print(f"Created {n_tiles} tiles of size {self.image_shape}x{self.image_shape}")
        
        # Process tiles in batches - ONLY BATCHES GO TO GPU
        output_tiles = []
        
        pbar = tqdm(range(0, n_tiles, self.batch_size), 
                   desc="Inference on tile batches", 
                   colour='#FFDB58')
        
        for tile_batch_start in pbar:
            # Load only this batch to GPU
            tile_batch = tiles[tile_batch_start:tile_batch_start+self.batch_size].to(self.device)
            
            with torch.inference_mode():
                pred = self.model(tile_batch)
            
            # Move predictions back to CPU to save GPU memory
            output_tiles.append(pred.cpu())
            
            # Clean up GPU memory
            del tile_batch, pred
            if self.device != 'cpu':
                torch.cuda.empty_cache()
        
        # Concatenate all tile predictions (on CPU)
        print("Reassembling tiles...")
        output_tiles = torch.cat(output_tiles, dim=0)
        
        # Refold tiles back into full image
        output_image = self._refold_with_blend(output_tiles, overlap=32)
        output_image = self._resize_output(output_image)
        
        # Initialize output
        label_image = np.zeros((self.in_channels, self.H, self.W), dtype=int)
        
        if return_transforms:
            transforms = output_image.squeeze(0)
        
        # Postprocess each compartment
        print("Postprocessing compartments...")
        for compartment in tqdm(range(self.in_channels), 
                               desc="Postprocessing", 
                               colour='#CE2029'):
            start_ind = int(compartment * self.n_compartment_predictions)
            end_ind = int((compartment+1) * self.n_compartment_predictions)
            
            # Extract compartment-specific channels and postprocess
            compartment_output = output_image[:, start_ind:end_ind]
            label_result = self._postprocess(compartment_output)
            label_image[compartment] = label_result.squeeze()

        print("Done segmenting image.")

        if return_transforms:
            return label_image, transforms
        else:
            return label_image
