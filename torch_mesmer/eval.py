from pathlib import Path
from datetime import datetime
import zarr
import numpy as np

from torch_mesmer.mesmer import Mesmer
from torch_mesmer.metrics import Metrics

from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from scipy.stats import hmean
import tqdm
import click

import pandas as pd

def pretty_print_summary(df: pd.DataFrame):

    report_summary = {
        'recall': 'Mean recall',
        'precision': 'Mean precision',
        'jaccard': 'Mean Jaccard index',
        'gained_detections': 'Mean gained detections',
        'missed_detections': 'Mean missed detections',
        'split': 'Mean split objects',
        'merge': 'Mean merged objects',
        'catastrophe': 'Mean many-to-many errors',
        'f1': 'Mean F1 score'
    }

    print('-'*10 + ' Summary ' + '-'*10)
    print()
    for col, col_name in report_summary.items():
        print(f'{col_name}:')
        print(f'    {df[col].mean():.2f}')

    return None

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


@click.command()
@click.option(
    '--device', 
    default='cpu', 
    help="""The device that you want to run the inference on. 
            Can be `'cpu'` or `'cuda'`. If `'cuda'`, can also specify the specific GPU if multiple are available.""")

@click.option(
    '--model-path', 
    default= Path.home() / ".deepcell/models/mesmer/saved_model_best_dict.pth", 
    help="""Path to model. 
            If unset, will use default DeepCell location (`~/.deepcell/models/mesmer/saved_model_best_dict.pth`)"""
            )

@click.option(
    '--data-path', 
    default=Path.home() / ".deepcell/tissuenet_v1-1/test.zarr" , 
    help="""Path to the Zarr file containing the test data for evaluation.
            If unset, defaults to DeepCell location (`~/.deepcell/tissuenet_v1-1/test.zarr)"""
        )

def main(device: str,
         model_path: str,
         data_path: str):
        
    z_test = zarr.open(data_path)

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

    y_test = np.flip(y_test, axis=-1)
    mpps = z_test['mpp'][:]

    X_test = np.moveaxis(X_test, 1,-1)
    y_test = np.moveaxis(y_test, 1,-1)
    y_test = np.flip(y_test, axis=-1)

    # Load model and application
    model = Mesmer(
        model_path=model_path,
        device=device,
    )

    compartments = ["w", "n"]

    for i in tqdm.tqdm(range(X_test.shape[0])):
        preds = model.predict(X_test[i:i+1], image_mpp=mpps[i], compartment="both")[0]
        for c, compartment in enumerate(compartments):
            metrics_out["compartment"].append(compartment)
            metrics = evaluate(preds[..., c], y_test[i:i+1, ..., c])
            for k, v in metrics.items():
                metrics_out[k].append(v)

    df = pd.DataFrame(metrics_out)

    pretty_print_summary(df)

    df.to_csv(f'eval_results_{datetime.now().isoformat(timespec="seconds")}.csv')
    return df

if __name__ == "__main__":
    main()
