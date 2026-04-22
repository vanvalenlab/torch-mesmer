from pathlib import Path
import os
import zarr
import glob
import numpy as np
import pandas as pd

def convert_to_zarr(filename, out_dir=None):

    if out_dir is None:
        file_dir = os.path.dirname(filename)
        split = os.path.splitext(os.path.basename(filename))[0]

    print(f"    Loading {split}.")
    data = np.load(os.path.join(filename), allow_pickle=True)
    X = data['X']
    y = data['y']
    meta = data['meta']

    # MPP is in the third column
    mpp = np.array(meta[:, 2])

    # coerce it into numerical and replace str and others with nan
    numeric_arr = pd.to_numeric(mpp, errors='coerce')

    # remove nan
    numeric_arr = numeric_arr[~np.isnan(numeric_arr)]
    numeric_arr = np.array(numeric_arr)

    # Make it channels first like PyTorch is expecting
    X = np.moveaxis(X, -1, 1)
    y = np.moveaxis(y, -1, 1)
    
    y = np.flip(y, axis=1)

    X = X[4:].astype(np.float32)
    y = y[4:].astype(np.float32)
    numeric_arr = numeric_arr[4:].astype(np.float32)

    B, C, H, W = X.shape

    # Create a Zarr store
    store = zarr.open(f"{file_dir}/{split}.zarr", mode="w")

    print(f"    Writing {split}.")

    # Store 'X' — chunked across C, H, W (one sample per chunk)
    store.create_dataset(
        "X",
        data=X,
        chunks=(1, C, H, W),  # chunk = one full image (all channels, full spatial dims)
        dtype=X.dtype,
    )

    # Store 'y' — chunked across C, H, W (one sample per chunk)
    store.create_dataset(
        "y",
        data=y,
        chunks=(1, C, H, W),
        dtype=y.dtype,
    )

    # Store 'mpp' — no spatial chunking needed, one scalar per sample
    store.create_dataset(
        "mpp",
        data=numeric_arr,
        chunks=(1,),
        dtype=numeric_arr.dtype,
    )


if __name__ == "__main__":

    data_directory = Path.home() / ".deepcell/tissuenet_v1-1/*.npz"

    if not (fnames := list(glob.glob(str(data_directory)))):
        raise ValueError("Tissuenet data not found at {data_directory}")

    
    for filename in fnames:
        print(f"Converting {os.path.basename(filename)}")
        convert_to_zarr(filename)
