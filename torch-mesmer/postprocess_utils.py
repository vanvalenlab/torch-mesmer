import torch
import numpy as np

from torch.nn import functional as F

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def get_gradients(transform, foreground_tensor, transform_thresh = 0.99):
    # Move to device and ensure correct dtypes

    transform = torch.where(foreground_tensor > transform_thresh, transform, -1).float()
    
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

def get_positions(transform):
    H, W = transform.shape

    # Initialize positions for all pixels
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=transform.device, dtype=torch.float32),
        torch.arange(W, device=transform.device, dtype=torch.float32),
        indexing='ij'
    )

    positions = torch.stack([y_coords, x_coords], dim=-1)  # (H, W, 2) -> [y, x]
    distances = torch.zeros((H, W))

    return positions, distances

def follow_flows(positions, gx, gy, niter=20, step_size=0.5):

    H, W = gx.squeeze().shape

    for step in range(niter):
        # Normalize positions to [-1, 1] for grid_sample
        norm_x = 2 * positions[..., 1] / (W - 1) - 1
        norm_y = 2 * positions[..., 0] / (H - 1) - 1
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
        positions[..., 0].clamp_(0, H - 1)
        positions[..., 1].clamp_(0, W - 1)    

    return positions

def one_step(positions, distances, gx, gy, step_size=0.5):
    
    H, W = gx.squeeze().shape

    # Normalize positions to [-1, 1] for grid_sample
    norm_x = 2 * positions[..., 1] / (W - 1) - 1
    norm_y = 2 * positions[..., 0] / (H - 1) - 1
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
    distances += torch.sqrt((step_size * gy_sample)**2 + (step_size * gx_sample)**2)

    positions[..., 0] = positions[..., 0] + step_size * gy_sample
    positions[..., 1] = positions[..., 1] + step_size * gx_sample


    # Clamp to image bounds
    positions[..., 0].clamp_(0, H - 1)
    positions[..., 1].clamp_(0, W - 1)    

    return positions, distances



class PositionAnimationBuilder:
    """
    Build an animation incrementally by adding frames during iteration.
    """
    
    def __init__(self, marker_size=50, title="Position Animation", 
                 xlim=None, ylim=None, figsize=(8, 6)):
        """
        Initialize the animation builder.
        
        Parameters:
        -----------
        marker_size : int
            Size of scatter markers
        title : str
            Plot title
        xlim : tuple, optional
            (min, max) for x-axis. Auto-determined if None
        ylim : tuple, optional
            (min, max) for y-axis. Auto-determined if None
        figsize : tuple
            Figure size (width, height)
        """
        self.frames = []  # Store (x_positions, y_positions) for each frame
        self.images = []
        self.distances = []
        self.marker_size = marker_size
        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.figsize = figsize
    
    def add_frame(self, x_positions, y_positions, image):
        """
        Add a new frame to the animation.
        
        Parameters:
        -----------
        x_positions : array-like
            X coordinates for this timestep
        y_positions : array-like
            Y coordinates for this timestep
        """
        x_positions = np.array(x_positions)
        y_positions = np.array(y_positions)
        self.frames.append((x_positions.copy(), y_positions.copy()))
        self.images.append(image.copy())
    
    def show(self, interval=50):
        """
        Generate and display the animation.
        
        Parameters:
        -----------
        interval : int
            Delay between frames in milliseconds
        
        Returns:
        --------
        HTML object with embedded animation
        """
        if not self.frames:
            print("No frames to animate!")
            return None
        
        # Determine axis limits if not provided
        if self.xlim is None:
            all_x = np.concatenate([frame[0] for frame in self.frames])
            x_min, x_max = all_x.min(), all_x.max()
            padding = 0.1 * (x_max - x_min) if x_max != x_min else 1
            xlim = (x_min - padding, x_max + padding)
        else:
            xlim = self.xlim
        
        if self.ylim is None:
            all_y = np.concatenate([frame[1] for frame in self.frames])
            y_min, y_max = all_y.min(), all_y.max()
            padding = 0.1 * (y_max - y_min) if y_max != y_min else 1
            ylim = (y_min - padding, y_max + padding)
        else:
            ylim = self.ylim
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(self.title)
        
        # Initialize scatter plot
        scatter = ax.scatter([], [], s=self.marker_size, alpha=0.6, c='w')
        imshow = ax.imshow(self.images[0])
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            return scatter, time_text
        
        def update(frame_idx):

            x_pos, y_pos = self.frames[frame_idx]
            curr_im = self.images[frame_idx]
            positions = np.column_stack((x_pos, y_pos))
            scatter.set_offsets(positions)
            imshow.set_array(curr_im)
            time_text.set_text(f'Frame: {frame_idx} ({len(x_pos)} points)')
            return scatter, time_text
        
        # Create animation
        anim = FuncAnimation(fig, update, init_func=init,
                            frames=len(self.frames), interval=interval,
                            blit=True, repeat=True)
        
        plt.close()  # Prevent static plot from displaying
        
        # Return as HTML5 video for inline display
        return HTML(anim.to_html5_video())
    
def merge_nearby_points(points, r):
    """
    Merge points within distance r by averaging coordinates.
    
    Parameters:
    -----------
    points : array-like, shape (n, 2)
        Array of (x, y) coordinates
    distance_matrix : array-like, shape (n, n)
        Pairwise distance matrix
    r : float
        Distance threshold for merging
    
    Returns:
    --------
    merged_points : ndarray, shape (m, 2)
        Array of merged point coordinates
    labels : ndarray, shape (n,)
        Cluster label for each original point
    """
    points = np.asarray(points)

    distance_matrix = cdist(points, points)    
    # Create adjacency matrix: points are connected if distance <= r
    adjacency = distance_matrix <= r
    
    # Find connected components
    n_components, labels = connected_components(
        csgraph=csr_matrix(adjacency),
        directed=False
    )
    
    # Merge points in each component by averaging
    merged_points = np.zeros((n_components, 2))
    for i in range(n_components):
        mask = labels == i
        merged_points[i] = points[mask].mean(axis=0)
    
    merged_points = merged_points.astype(int)
    
    return merged_points