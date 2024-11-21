"""
================================
Predict: Whole-cell Segmentation
================================
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import tifffile as tff
import napari

import torch
from torch_mesmer.model_utils import create_model
from torch_mesmer.mesmer import Mesmer

deepcell_path = Path.home() / ".deepcell"
# TODO - update model name when released
model_path = deepcell_path / "models/saved_model_full_8_best_dict.pth"

model, _, _ = create_model()
model.load_state_dict(
    torch.load(model_path, map_location="cpu", weights_only=True)
)

# Convert to CPU for CI
model.to("cpu")
model.eval()

app = Mesmer(model=model)

# Acquire sample data
# TODO: Switch to public zarr
# from deepcell.datasets import TissueNetSample
# X, _ = TissueNetSample().load_data()
img_fname = deepcell_path / "datasets/tissuenet-sample.npz"
img = np.load(img_fname)["X"]
print(f"Image shape: {img.shape}")

# TODO: Check image_mpp
mask = app.predict(img)

# Visualize image
nim = napari.view_image(img, channel_axis=-1, name=["nuclear", "membrane"])
# Construct a binary colormap for visualizing cell boundaries
color_dict = {k: np.ones(3) for k in range(1, mask.max() + 1)}
color_dict[None] = None
binary_colormap = napari.utils.DirectLabelColormap(color_dict=color_dict)
# Visualize segmentation
labels_layer = nim.add_labels(
    mask.squeeze(), name="segmentation", colormap=binary_colormap, opacity=1
)
labels_layer.contour = 1  # Show label edges instead of mask

# Start event loop and show the viewer
if __name__ == "__main__":
    napari.run()
