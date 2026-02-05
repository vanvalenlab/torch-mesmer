import os
import zarr
import glob
import numpy as np

def convert_to_zarr(filename, out_dir=None):

    if out_dir is None:
        file_dir = os.path.dirname(filename)
        split = os.path.splitext(os.path.basename(filename))[0]
        output_file = os.path.join(file_dir, split) + '.zarr'

    data = np.load(os.path.join(filename))
    X = data['X']
    y = data['y']


    # Make it channels first like PyTorch is expecting
    X = np.moveaxis(X, -1, 1)
    y = np.moveaxis(y, -1, 1)

    z = zarr.open(output_file, 'w')

    z.array(name='X', data=X, shape=X.shape, chunks=(1, X.shape[1], X.shape[2], X.shape[3]))
    z.array(name='y', data=y, shape=X.shape, chunks=(1, X.shape[1], X.shape[2], X.shape[3]))


if __name__ == "__main__":

    data_directory = '/data/shared/tissuenet/*.npz'
    
    for filename in glob.glob(data_directory):
        print(f"Converting {os.path.basename(filename)}")
        convert_to_zarr(filename)