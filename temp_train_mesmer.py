import argparse
import os
import tempfile

import numpy as np
# import tensorflow as tf
# from deepcell import image_generators, losses
# from deepcell.applications.mesmer import mesmer_preprocess
# from deepcell.model_zoo.panopticnet import PanopticNet
# from deepcell.utils.train_utils import rate_scheduler
# from dvc.utils.serialize import dump_yaml
# from tensorflow.keras.losses import MSE
# from tensorflow.keras.optimizers import Adam

MODEL_DIR = "."
DATA_DIR = "../../training-data"


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        default=os.path.join(MODEL_DIR, "mesmer/MultiplexSegmentation"),
        help="Destination of the trained model.",
    )

    parser.add_argument(
        "--metrics-path",
        default=os.path.join(MODEL_DIR, "mesmer/train-metrics.yaml"),
        help="Destination of recorded metrics of the trained model.",
    )

    parser.add_argument(
        "--data-path",
        default=os.path.join(DATA_DIR, "training-data/tissue-net"),
        help="Path to the training data.",
    )

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--min-objects",
        type=int,
        default=0,
        help="Minimum number of objects in each training image.",
    )

    parser.add_argument(
        "--zoom-min",
        type=float,
        default=0.7,
        help="Smallest zoom value. Zoom max is inverse of zoom min.",
    )

    parser.add_argument(
        "--batch-size", type=int, default=8, help="Number of samples per batch."
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Backbone of the model to train.",
    )

    parser.add_argument(
        "--crop-size", type=int, default=256, help="Size of square patches to train on."
    )

    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Size of square patches to train on."
    )

    return parser


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


def load_data(filepath):
    """Load train, val, and test data"""
    X_train, y_train = _load_npz(os.path.join(filepath, "train.npz"))

    X_val, y_val = _load_npz(os.path.join(filepath, "val_256x256.npz"))

    return (X_train, y_train), (X_val, y_val)


def semantic_loss(n_classes):
    def _semantic_loss(y_pred, y_true):
        if n_classes > 1:
            return 0.01 * losses.weighted_categorical_crossentropy(
                y_pred, y_true, n_classes=n_classes
            )
        return MSE(y_pred, y_true)

    return _semantic_loss


def create_model(input_shape=(256, 256, 2), backbone="resnet50", lr=1e-4):
    model = PanopticNet(
        backbone=backbone,
        input_shape=input_shape,
        norm_method=None,
        num_semantic_heads=4,
        num_semantic_classes=[
            1,
            3,
            1,
            3,
        ],  # inner distance, pixelwise, inner distance, pixelwise
        location=True,  # should always be true
        include_top=True,
    )

    loss = {}

    # Give losses for all of the semantic heads
    for layer in model.layers:
        if layer.name.startswith("semantic_"):
            n_classes = layer.output_shape[-1]
            loss[layer.name] = semantic_loss(n_classes)

    optimizer = Adam(lr=lr, clipnorm=0.001)

    model.compile(loss=loss, optimizer=optimizer)

    return model


def create_prediction_model(input_shape, backbone, weights_path):
    """Create version of model without custom losses"""
    prediction_model = PanopticNet(
        backbone=backbone,
        input_shape=input_shape,
        norm_method=None,
        num_semantic_heads=4,
        num_semantic_classes=[
            1,
            3,
            1,
            3,
        ],  # inner distance, pixelweise, inner distance, pixelwise
        location=True,  # should always be true
        include_top=True,
    )
    prediction_model.load_weights(weights_path, by_name=True)
    return prediction_model


def create_data_generators(
    train_dict,
    val_dict,
    rotation_range=180,
    shear_range=0,
    zoom_min=0.7,
    horizontal_flip=True,
    vertical_flip=True,
    crop_size=(256, 256),
    seed=0,
    batch_size=8,
    min_objects=0,
):
    # use augmentation for training but not validation
    datagen = image_generators.CroppingDataGenerator(
        rotation_range=rotation_range,
        shear_range=shear_range,
        zoom_range=(zoom_min, 1 / zoom_min),
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        crop_size=(crop_size, crop_size),
    )

    datagen_val = image_generators.SemanticDataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=0,
        vertical_flip=0,
    )

    train_data = datagen.flow(
        train_dict,
        seed=seed,
        transforms=["inner-distance", "pixelwise"],
        transforms_kwargs={
            "pixelwise": {"dilation_radius": 1},
            "inner-distance": {"erosion_width": 1, "alpha": "auto"},
        },
        min_objects=0,
        batch_size=batch_size,
    )

    val_data = datagen_val.flow(
        val_dict,
        seed=seed,
        transforms=["inner-distance", "pixelwise"],
        transforms_kwargs={
            "pixelwise": {"dilation_radius": 1},
            "inner-distance": {"erosion_width": 1, "alpha": "auto"},
        },
        min_objects=min_objects,
        batch_size=batch_size,
    )

    return train_data, val_data


def train(model, train_data, val_data, model_path, lr=1e-4, n_epoch=100, batch_size=8):
    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    monitor = "val_loss"
    rate_scheduler(lr=lr, decay=0.99)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
            save_weights_only=False,
        ),
        tf.keras.callbacks.LearningRateScheduler(rate_scheduler(lr=lr, decay=0.99)),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.33,
            patience=5,
            verbose=1,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        ),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    ]

    history = model.fit(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=callbacks,
    )

    return history


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    # convert paths to be relative path from this file
    model_path = os.path.join(os.path.dirname(__file__), args.model_path)
    metrics_path = os.path.join(os.path.dirname(__file__), args.metrics_path)
    data_path = os.path.join(os.path.dirname(__file__), args.data_path)

    # load the data
    (X_train, y_train), (X_val, y_val) = load_data(args.data_path)

    train_dict = {"X": mesmer_preprocess(X_train), "y": y_train}
    val_dict = {"X": mesmer_preprocess(X_val), "y": y_val}

    # instantiate model
    model = create_model(
        input_shape=(args.crop_size, args.crop_size, 2),
        backbone=args.backbone,
        lr=args.lr,
    )

    # create data generators
    train_data, val_data = create_data_generators(
        train_dict,
        val_dict,
        seed=args.seed,
        zoom_min=args.zoom_min,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
    )

    loss_history = train(
        model, train_data, val_data, args.model_path, args.lr, args.epochs
    )

    all_metrics = {"metrics": {k: str(v[-1]) for k, v in loss_history.history.items()}}

    # save a metadata.yaml file in the saved model directory, as well as weights for evaluate.py
    dump_yaml(args.metrics_path, all_metrics)
    # Reload the model for prediction
    with tempfile.TemporaryDirectory() as tmpdirname:
        weights_path = os.path.join(str(tmpdirname), "model_weights.h5")
        model.save_weights(weights_path, save_format="h5")
        prediction_model = create_prediction_model(
            input_shape=(args.crop_size, args.crop_size, 2),
            backbone=args.backbone,
            weights_path=weights_path,
        )
        prediction_model.save(model_path, include_optimizer=False, overwrite=True)