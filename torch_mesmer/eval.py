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
        'data_path': Path.home() / ".deepcell/tissuenet_v1-1/test.zarr",
        'model_path': Path.home() / ".deepcell/models/mesmer/saved_model_best_dict_e150dafc.pth",
        'device': "cuda:1",
    }

    # Whole cell, nuc
        
    z_test = zarr.open(f"{config['data_path']}")

    # current X is in shape B, C, H, W, needs to be in B, H, W, C

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

    X_test = np.moveaxis(z_test['X'][:], 1, -1)
    y_test = np.moveaxis(z_test['y'][:], 1, -1)
    mpps = z_test['mpp'][:]

    # Load model and application
    model = Mesmer(
        model_path = config['model_path'],
        device=config["device"],
    )

    compartments = ['w','n']


    for i in tqdm.tqdm(range(X_test.shape[0])):

        preds = model.predict(X_test[i:i+1], image_mpp=mpps[i], compartment='both')[0]

        for c, compartment in enumerate(compartments):
            metrics_out["compartment"].append(compartment)
            metrics = evaluate(preds[...,c], y_test[i:i+1,...,c])
            for k, v in metrics.items():
                metrics_out[k].append(v)

    df = pd.DataFrame(metrics_out)
    df.to_csv('eval_results_mesmer_hybrid.csv')

if __name__ == "__main__":
    main()
