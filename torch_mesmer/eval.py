import zarr
import numpy as np

from torch_mesmer.mesmer import Mesmer
from torch_mesmer.metrics import Metrics

from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from scipy.stats import hmean
import tqdm

import pandas as pd

def evaluate(y_pred, y_test):
    m = Metrics("DVC Mesmer")
    metrics = m.calc_object_stats(y_test, y_pred, progbar=False)

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
        'model_path': 'data/model/20260213141132/saved_model_best_dict.pth'
    }

    # Whole cell, nuc
        
    z_test = zarr.open(f"{config['data_path']}")

    metrics_out = {
        "recall": [],
        "precision": [],
        "jaccard": [],
        "n_true": [],
        "n_pred": [],
        "gained_detections": [],
        "missed_detections": [],
        "split": [],
        "merge": [],
        "catastrophe": [],
        "f1": [],
        "compartment": []
    }

    X_test = z_test['X'][:]
    y_test = z_test['y'][:]
    mpps = z_test['mpp'][:]

    # Load model and application
    model = Mesmer(
        model_path = config['model_path'],
        device='cuda:2',
    )

    compartments = ['n','w']

    preds = model.segment(X_test, mpps=mpps, postprocess_method='hybrid')

    for i in tqdm.tqdm(range(preds.shape[0])):
        for c, compartment in enumerate(compartments):
            metrics_out["compartment"].append(compartment)
            metrics = evaluate(preds[i:i+1,c], y_test[i:i+1,c])
            for k, v in metrics.items():
                metrics_out[k].append(v)

    df = pd.DataFrame(metrics_out)
    df.to_csv('eval_results_mesmer_hybrid.csv')

if __name__ == "__main__":
    main()
