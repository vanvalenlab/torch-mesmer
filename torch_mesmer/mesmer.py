import torch
from torch.nn import functional as F
from torchvision.transforms import functional as fvision

import numpy as np
from tqdm import tqdm

from torch_mesmer.utils import histogram_normalization, percentile_threshold, deep_watershed
from torch_mesmer.model import PanopticNet
import math
import skimage
from torch_mesmer.postprocess_utils import merge_nearby_points
from skimage.measure import regionprops


class Mesmer():

    def __init__(
            self, 
            model_path=None, 
            device=None, 
            batch_size = 16,
            data_format = 'channels_first',
            postprocess_method = 'hybrid'
    ):

        if device is None:
            self.device = 'cpu'
        else:
            self.device=device

        if model_path is None:
            raise Exception("Please provide a path to the model checkpoint file.")
        
        print("Initializing model...")
        self.model = PanopticNet(
            crop_size=256,
            backbone='resnet50',
            pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
            backbone_levels=['C3', 'C4', 'C5'],
            n_semantic_classes=[1,3,1,3]
        ).to(self.device).eval()

        # Dummy data to make semantic heads
        dummy = torch.rand(1, 2, self.model.crop_size, self.model.crop_size).to(self.device)
        _ = self.model(dummy)
        del dummy

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        print(f"Model initialized. \n   Using device: {device}")
        print()
        # Whole cell, nuc
        self.postprocess_kwargs = [{
            'maxima_threshold': 0.075,
            'maxima_smooth': 0,
            'interior_threshold': 0.2,
            'interior_smooth': 2,
            'small_objects_threshold': 15,
            'fill_holes_threshold': 15,
            'radius': 2
        },
        {
            'maxima_threshold': 0.1,
            'maxima_smooth': 0,
            'interior_threshold': 0.2,
            'interior_smooth': 2,
            'small_objects_threshold': 15,
            'fill_holes_threshold': 15,
            'radius': 2
        }]

        self.device = device
        self.batch_size = batch_size
        self.data_format = data_format

        self.image_shape = self.model.crop_size
        self.in_channels = 2
        self.out_channels = 8

        self.model_mpp = 0.5
            
        self.n_iter = 40
        self.step_size = 0.1
        self.postprocess_method = postprocess_method
        self.transform_thresh = 0.05
        self.relevant_votes = 20
        self.radius = 12

        self.n_compartment_predictions = self.out_channels / self.in_channels

    def _preprocess(self, x):

        assert len(x.shape) == 4, 'add batch dimension'

        x = percentile_threshold(x, percentile=99.9)
        x = histogram_normalization(x, data_format=self.data_format)

        return x

    def _resize_input(self, x, resize_factor):

        upscale_H = self.H * resize_factor
        upscale_W = self.W * resize_factor
        new_size = (int(upscale_H), int(upscale_W))

        x = fvision.resize(x, new_size, interpolation = fvision.InterpolationMode.BILINEAR)

        # Pad if smaller than model crop size
        _, _, h, w = x.shape
        if h < self.image_shape or w < self.image_shape:
            pad_h = max(0, self.image_shape - h)
            pad_w = max(0, self.image_shape - w)
            
            # Pad symmetrically (left, right, top, bottom)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            
            # Store padding info for later cropping
            self.pad_left = pad_left
            self.pad_right = pad_right
            self.pad_top = pad_top
            self.pad_bottom = pad_bottom
            self.padded_h = h
            self.padded_w = w
        else:
            self.pad_left = 0
            self.pad_right = 0
            self.pad_top = 0
            self.pad_bottom = 0
            self.padded_h = h
            self.padded_w = w

        return x
    
    def _unfold_with_overlap(self, x, overlap=32):
        """
        Extract overlapping tiles from image sequence
        image: (T, C, H, W) tensor
        Returns tiles of shape (T*n_tiles_h*n_tiles_w, C, tile_size, tile_size)
        """
        self.curr_batch_size, C, H, W = x.shape

        # Special case: if image is exactly crop size or smaller, return as single tile
        if H <= self.image_shape and W <= self.image_shape:
            self.nh = 1
            self.nw = 1
            return x  # (T, C, H, W) - already the right size
        
        stride = self.image_shape - overlap
        
        # Calculate number of tiles needed - ensure we cover the entire image
        nh = int(np.ceil((H - self.image_shape) / stride)) + 1
        nw = int(np.ceil((W - self.image_shape) / stride)) + 1
        
        # Store for later use in refold
        self.nh = nh
        self.nw = nw
        
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
        tiles: (T*n_tiles_h*n_tiles_w, C, tile_size, tile_size) or (T, C, H, W) for single tile
        original_shape: (T, C, H, W)
        """
        T = self.curr_batch_size
        C = self.out_channels

        # Special case: single tile (image was smaller than or equal to crop size)
        if self.nh == 1 and self.nw == 1:
            # tiles is already (T, C, H, W) - just return it
            return tiles
        
        stride = self.image_shape - overlap

        # Use padded dimensions for reconstruction
        upscale_H = self.padded_h
        upscale_W = self.padded_w
        
        # Use stored tile counts
        nh = self.nh
        nw = self.nw
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
    
    def _resize_output(self, x):
        
        # Remove padding first if it was added
        if self.pad_top > 0 or self.pad_bottom > 0 or self.pad_left > 0 or self.pad_right > 0:
            _, _, h, w = x.shape
            x = x[:, :, 
                  self.pad_top:h-self.pad_bottom if self.pad_bottom > 0 else h,
                  self.pad_left:w-self.pad_right if self.pad_right > 0 else w]
        
        # Resize back to original dimensions
        x = fvision.resize(x, (self.H, self.W), interpolation=fvision.InterpolationMode.BILINEAR)

        return x
    

    def _get_gradients(self, transform, foreground_tensor):
        # Move to device and ensure correct dtypes

        transform = torch.where(foreground_tensor > self.transform_thresh, transform, -1).float()
        
        transform_ds = transform.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Compute gradients of INNER distance (points toward peaks)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=torch.float32, device=transform.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=torch.float32, device=transform.device).view(1, 1, 3, 3)
        
        gx = F.conv2d(transform_ds, sobel_x, padding=1)  # (1, 1, H, W) or (1, 1, H//ds, W//ds)
        gy = F.conv2d(transform_ds, sobel_y, padding=1)  # (1, 1, H, W) or (1, 1, H//ds, W//ds)

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
    
    def _postprocess(self, x, kwargs, markers = None, resize_factor = 1):

        label_image = np.zeros((self.curr_batch_size, 1, self.H, self.W), dtype=int)
        marker_image = np.zeros((self.curr_batch_size, 1, self.H, self.W), dtype=int)

        for t in range(self.curr_batch_size):

            x_inner = x[t, 0].cpu().numpy()
            x_peri = x[t, 1].cpu().numpy()
            x_foreground = x[t, 2].cpu().numpy()
            x_background = x[t, 3].cpu().numpy()

            if self.postprocess_method == 'hybrid':

                # Downsample factor - increase for larger images to save memory
                # For a 2048x2048 image, downsample_factor=4 gives you a 512x512 grid
                downsample_factor = max(1, min(self.H, self.W) // 4096)
                
                # Use the same downsample factor for both positions and gradients
                positions = self._get_positions(x[t, 0], downsample_factor=downsample_factor)
                gx, gy = self._get_gradients(x[t, 0], x[t, 2])
                positions = self._follow_flows(positions, gx, gy, niter=self.n_iter, step_size=self.step_size)
                
                # Flatten positions to get all converged points: (N, 2) where N = H_ds * W_ds
                converged_points = positions.reshape(-1, 2).cpu().numpy().astype(int)
                
                # Clip to image bounds (in case of any edge artifacts)
                converged_points[:, 0] = np.clip(converged_points[:, 0], 0, self.H - 1)
                converged_points[:, 1] = np.clip(converged_points[:, 1], 0, self.W - 1)
                
                # Sample background at converged points to filter out background regions
                background_values = x[t, 3].cpu().numpy()[converged_points[:, 0], converged_points[:, 1]]
                
                # Keep only points that converged to foreground (low background value)
                foreground_mask = background_values < self.transform_thresh
                relevant_points = converged_points[foreground_mask]
                
                # Count how many starting points converged to each location
                unique_points, counts = np.unique(relevant_points, axis=0, return_counts=True)
                
                # Filter by count threshold (only keep peaks with enough votes)
                high_count_mask = counts > self.relevant_votes/downsample_factor
                centroid_candidates = unique_points[high_count_mask]
                
                # Merge nearby centroids (avoid duplicates from nearby convergence)
                final_centroids = merge_nearby_points(centroid_candidates, r=self.radius * resize_factor)
                
                # Create markers image with one-pixel dots at centroid locations
                markers = np.zeros((self.H, self.W), dtype=np.uint8)
                if len(final_centroids) > 0:
                    markers[final_centroids[:, 0], final_centroids[:, 1]] = 1

                markers = skimage.measure.label(markers)
                label_image[t], marker_image[t] = deep_watershed(x_inner, x_foreground, markers, kwargs=kwargs)

            elif self.postprocess_method == 'classical':
                label_image[t], marker_image[t] = deep_watershed(x_inner, x_foreground, markers, kwargs=kwargs)

            
        label_image = label_image.astype(int)

        return label_image, marker_image
        
    def segment(self,
                x,
                mpps = None,
                data_format='channels_first',
                return_transforms = False,
                return_markers = False,
                verbose=False):

        if data_format == 'channels_last':
            x = np.moveaxis(x, -1, 1)

        # Assume single image input - add batch dimension if needed
        if len(x.shape) == 3:
            x = x[np.newaxis, ...]
        
        # Keep track of original shape
        self.H = x.shape[-2]
        self.W = x.shape[-1]
        n_frames = x.shape[0]
        self.curr_batch_size = 1  # Processing single image
        
        label_image = np.zeros((n_frames, self.in_channels, self.H, self.W), dtype=int)
        transforms = np.zeros((n_frames, self.out_channels, self.H, self.W), dtype=np.float32)
        markers = np.zeros((n_frames, self.in_channels, self.H, self.W), dtype=int)

        if mpps is None:
            resize_factors = np.ones((n_frames,))
        elif isinstance(mpps, (list, np.ndarray)):
            resize_factors = mpps/self.model_mpp
        else:
            resize_factors = np.full((n_frames), mpps/self.model_mpp)

        pbar = tqdm(range(n_frames), 
                desc="Processing", leave=False)
        
        for t in pbar:
            # Preprocess the image (on CPU)
            if verbose:
                pbar.set_description("Preprocessing image...")

            x_batch = x[t]
            resize_factor = resize_factors[t]

            if len(x_batch.shape) == 3:
                x_batch = x_batch[np.newaxis, ...]

            x_batch = self._preprocess(x_batch)

            # Convert to tensor but keep on CPU for now
            x_tensor = torch.from_numpy(x_batch)
            
            # Resize on CPU (and pad if necessary)
            if verbose:
                pbar.set_description("Resizing image...")
            x_resized = self._resize_input(x_tensor, resize_factor)
            
            # Unfold into tiles (on CPU) - this doesn't load to GPU yet
            if verbose:
                pbar.set_description("Tiling image...")
            tiles = self._unfold_with_overlap(x_resized, overlap=32)
            tiles = tiles.float()
            
            n_tiles = tiles.shape[0]
            
            # Process tiles in batches - ONLY BATCHES GO TO GPU
            output_tiles = []
            
            for tile_batch_start in range(0, n_tiles, self.batch_size):
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
            if verbose:
                pbar.set_description("Reassembling image...")
            output_tiles = torch.cat(output_tiles, dim=0)
            
            # Refold tiles back into full image
            output_image = self._refold_with_blend(output_tiles, overlap=32)
            
            # Resize back to original size (removes padding if it was added)
            output_image = self._resize_output(output_image)

            if return_transforms:
                transforms[t] = output_image.squeeze(0).cpu().numpy()

            # Postprocess each compartment
            if verbose:
                pbar.set_description("Postprocessing compartments...")

            for compartment in range(self.in_channels):
                start_ind = int(compartment * self.n_compartment_predictions)
                end_ind = int((compartment+1) * self.n_compartment_predictions)
                
                # Extract compartment-specific channels and postprocess
                compartment_output = output_image[:, start_ind:end_ind]
                label_result, marker_result = self._postprocess(compartment_output, self.postprocess_kwargs[compartment], resize_factor=resize_factor)
                label_image[t, compartment] = label_result.squeeze()
                markers[t, compartment] = marker_result.squeeze()

        print("Done segmenting image.")

        if return_transforms:
            return label_image, transforms
        if return_markers:
            return label_image, markers
        else:
            return label_image
