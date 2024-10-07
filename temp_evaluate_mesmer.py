import argparse
import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from deepcell.applications import Mesmer
from deepcell_toolbox.metrics import Metrics
from dvc.utils.serialize import dump_yaml
from scipy.stats import hmean

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
        "--evaluate-metrics-path",
        default=os.path.join(MODEL_DIR, "mesmer/evaluate-metrics.yaml"),
        help="Destination of recorded metrics of the trained model.",
    )

    parser.add_argument(
        "--data-path",
        default=os.path.join(DATA_DIR, "tissue-net"),
        help="Path to the training data.",
    )

    parser.add_argument(
        "--seed", default=0, help="Random seed to make dataset splitting reproducible."
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


def evaluate(y_pred, y_test):
    m = Metrics("DVC Mesmer")
    metrics = m.calc_object_stats(y_test, y_pred)

    # calculate image-level recall and precision for F1 score
    recall = metrics["correct_detections"].values / metrics["n_true"].values
    recall = np.where(np.isfinite(recall), recall, 0)

    precision = metrics["correct_detections"] / metrics["n_pred"]
    precision = np.where(np.isfinite(precision), precision, 0)
    f1 = hmean([recall, precision])

    # record summary stats
    summary = m.summarize_object_metrics_df(metrics)

    valid_keys = {
        "recall",
        "precision",
        "jaccard",
        "n_true",
        "n_pred",
        "gained_detections",
        "missed_detections",
        "split",
        "merge",
        "catastrophe",
    }

    output_data = {}
    for k in valid_keys:
        if k in {"jaccard", "recall", "precision"}:
            output_data[k] = float(summary[k])
        else:
            output_data[k] = int(summary[k])
    output_data["f1"] = float(np.mean(f1))

    return output_data


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    # convert paths to be relative path from this file
    model_path = os.path.join(os.path.dirname(__file__), args.model_path)
    metrics_path = os.path.join(os.path.dirname(__file__), args.evaluate_metrics_path)
    data_path = os.path.join(os.path.dirname(__file__), args.data_path)

    X_test, y_test = _load_npz(os.path.join(data_path, "test_256x256.npz"))

    # Load saved model and initialize application
    model = tf.keras.models.load_model(model_path)
    app = Mesmer(model)

    # evaluate the model
    cell_preds = app.predict(X_test)
    cell_metrics = evaluate(cell_preds, y_test[..., :1])

    nuc_preds = app.predict(X_test, compartment="nuclear")
    nuc_metrics = evaluate(nuc_preds, y_test[..., 1:])

    combined_metrics = {
        "cell_metrics": OrderedDict(sorted(cell_metrics.items())),
        "nuc_metrics": OrderedDict(sorted(nuc_metrics.items())),
    }

    dump_yaml(metrics_path, combined_metrics)