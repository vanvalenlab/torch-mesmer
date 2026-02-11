from torch_mesmer.mesmer import Mesmer

import matplotlib.pyplot as plt
import numpy as np

import zarr
from torch.utils.tensorboard import SummaryWriter

from skimage.color import label2rgb
from skimage.exposure import rescale_intensity

from torch_mesmer.metrics import Metrics
from scipy.stats import hmean
import numpy as np

import pandas as pd

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


def create_overlays(x, gt, pred):
    x = np.squeeze(x)
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)

    # Rescale raw data
    percentiles = np.percentile(x[np.nonzero(x)], [5, 95])
    raw = rescale_intensity(
        x, in_range=(percentiles[0], percentiles[1]), out_range="float32"
    )

    # Overlay gt on raw
    gt_overlay = label2rgb(gt, image=raw, bg_label=0)

    # Overlay pred on raw
    pred_overlay = label2rgb(pred, image=raw, bg_label=0)

    return gt_overlay, pred_overlay

def main():

    config = {
        'eval_info': 'data/segmentation/eval',
        'data_path': '/data/shared/tissuenet/tissuenet_v1.1_test.zarr',
        'model_path': 'data/model/20260210174005/saved_model_best_dict.pth'
    }

    # Whole cell, nuc
    postprocess_kwargs = [{
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
        
    z_test = zarr.open(f"{config['data_path']}")

    X_test = z_test['X'][:]
    y_test = z_test['y'][:]
    mpps = z_test['mpp'][:]

    # Load model and application
    model = Mesmer(
        model_path = config['model_path'],
        device='cuda:1',
        postprocess_method='hybrid'
    )

    # evaluate the model
    # TODO: evaluate based on experiment data type

    preds = model.segment(X_test, mpps=mpps)
    
    print('Calculating metrics...')
    for c in range(X_test.shape[1]):
        evaluate(preds[:,c], y_test[:,c])


if __name__ == "__main__":
    main()
