import os
import numpy as np

def load_data(filepath):
    """Load train, val, and test data"""
    X_train, y_train = _load_npz(os.path.join(filepath, "train.npz"))

    X_val, y_val = _load_npz(os.path.join(filepath, "val_256x256.npz"))

    return (X_train, y_train), (X_val, y_val)

def _load_npz(filepath):
    """Load a npz file"""
    data = np.load(filepath)
    X = data["X"]
    y = data["y"]

    print(
        "Loaded {}: X.shape: {}, y.shape {}".format(
            os.path.basename(filepath), X.shape, y.shape
        )
    )

    return X, y