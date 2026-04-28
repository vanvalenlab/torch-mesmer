from pathlib import Path
import os
import zarr
import glob
import numpy as np
import pandas as pd

def convert_to_zarr(filename, out_dir=None):

    # there are four very poorly segmented images in the beginnign of the test split
    # remove those
    split_custom_crop = {'test': 8, 'train': 1, 'val': 1}

    if out_dir is None:
        file_dir = os.path.dirname(filename)
        split = os.path.splitext(os.path.basename(filename))[0]

    print(f"    Loading {split}.")
    data = np.load(os.path.join(filename), allow_pickle=True)
    X = data['X']
    y = data['y']

    crop_val = split_custom_crop[split]
    offset = 0
    
    ## header is repeated 3 additional times in test split 
    #throwing off cropping X and y
    if split == 'test':
        offset = 3 

    # Make it channels first like PyTorch is expecting
    X = np.moveaxis(X, -1, 1)
    y = np.moveaxis(y, -1, 1)
    
    y = np.flip(y, axis=1) # correct channels once and only once for Y

    X = X[crop_val-1-offset:].astype(np.float32)
    y = y[crop_val-1-offset:].astype(np.float32)

    B, C, H, W = X.shape

    # Create a Zarr store
    store = zarr.open(f"{file_dir}/{split}.zarr", mode="w")

    print(f"    Writing {split}.")

    # Store 'X' — chunked across C, H, W (one sample per chunk)
    store.create_array(
        "X",
        data=X,
        chunks=(1, C, H, W),  # chunk = one full image (all channels, full spatial dims)
    )

    # Store 'y' — chunked across C, H, W (one sample per chunk)
    store.create_array(
        "y",
        data=y,
        chunks=(1, C, H, W),
    )

    meta_dtype = np.dtype(
    [("filename", "U128"), ("experiment", "U128"), ("pixel_size", float), ("specimen", "U128")]
    )

    meta_ary = np.zeros(len(data["meta"]) - crop_val, dtype=meta_dtype)
    meta_ary["filename"][:] = data["meta"][crop_val:, 0]
    meta_ary["experiment"] = data["meta"][crop_val:, 1]
    meta_ary["pixel_size"] = data["meta"][crop_val:, 2]
    meta_ary["specimen"] = data["meta"][crop_val:, -1]

    # Store 'metadata' — no spatial chunking needed, one scalar per sample
    store.create_array(
        "meta",
        data=meta_ary,
    )

if __name__ == "__main__":

    data_directory = Path.home() / ".deepcell/tissuenet_v1-1/*.npz"

    if not (fnames := list(glob.glob(str(data_directory)))):
        raise ValueError("Tissuenet data not found at {data_directory}")

    for filename in fnames:
        print(f"Converting {os.path.basename(filename)}")
        convert_to_zarr(filename)
