import os
import zarr
import glob
import numpy as np
import pandas as pd

def convert_to_zarr(filename, out_dir=None):

    if out_dir is None:
        file_dir = os.path.dirname(filename)
        split = os.path.splitext(os.path.basename(filename))[0]
        output_file = os.path.join(file_dir, split) + '.zarr'

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
    
    print(X.shape)
    print(y.shape)
    print(numeric_arr.shape)

    # Make it channels first like PyTorch is expecting
    X = np.moveaxis(X, -1, 1)
    y = np.moveaxis(y, -1, 1)

    y = np.flip(y, axis=1)

    X = X[4:]
    y = y[4:]
    numeric_arr = numeric_arr[4:]

    z = zarr.open(output_file, mode='w')

    z.create_array(name='X', data=X, chunks=(1, X.shape[1], X.shape[2], X.shape[3]))
    z.create_array(name='y', data=y, chunks=(1, X.shape[1], X.shape[2], X.shape[3]))
    z.create_array(name='mpp', data=numeric_arr)


if __name__ == "__main__":

    data_directory = '/data/shared/tissuenet/*.npz'
    
    for filename in glob.glob(data_directory):
        print(f"Converting {os.path.basename(filename)}")
        convert_to_zarr(filename)