from dnn import DNN

import matplotlib.pyplot as plt
import numpy as np

import zarr
from torch.utils.tensorboard import SummaryWriter

from skimage.color import label2rgb
from skimage.exposure import rescale_intensity

from metrics import Metrics
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
        'data_path': 'data/DynamicNuclearNet-segmentation-v1_0',
        'model_path': 'data/segmentation/model/20260126142939/saved_model_best_dict.pth'
    }

    postprocess_kwargs = {
                'radius': 10,
                'maxima_threshold': 0.05,
                'reduced_thresh': 0.05,
                'transform_thresh': 0.0,
                'n_iter': 100,
                'step_size': 0.5,
                'eccentricity': 1.0,
                'postprocess_method': 'hybrid',
                'relevant_counts': 100,
                'small_objects_threshold': 0
            }
        
    # writer = SummaryWriter(config['eval_info'])

    z_test = zarr.open(f"{config['data_path']}/test.zarr")

    meta_test = pd.read_json(f"{config['data_path']}/test.json")

    test_mpps = meta_test['pixel_size'].to_numpy()

    X_test = z_test['X'][:]
    y_test = z_test['y'][:]

    if test_mpps is not None:
        good_mpps = ~np.isnan(test_mpps)

        X_test = X_test[good_mpps]
        y_test = y_test[good_mpps]
        test_mpps = test_mpps[good_mpps]

    y_test = np.moveaxis(y_test, -1, 1)

    # Load model and application
    model = DNN(
        model_path = config['model_path'],
        device='cuda:1',
        postprocess_kwargs=postprocess_kwargs
    )

    # evaluate the model
    # TODO: evaluate based on experiment data type
    preds = model.segment(X_test, data_format='channels_last')
    metrics = evaluate(preds, y_test)

    # writer.close()

if __name__ == "__main__":
    main()
