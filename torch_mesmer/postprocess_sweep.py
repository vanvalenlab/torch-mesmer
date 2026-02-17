import zarr
import numpy as np
import itertools
import random

from torch_mesmer.mesmer import Mesmer
from torch_mesmer.metrics import Metrics

from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from scipy.stats import hmean
from tqdm import tqdm

import pandas as pd


def make_sweep(sweep, default_kwargs, sweep_type):

    default_kwargs['postprocess_method'] = sweep_type

    sweep_list = []

    sweep_names = [*sweep.keys()]
    sweep_vals = [*sweep.values()]

    for combos in itertools.product(*sweep_vals):
        curr_kwargs = default_kwargs.copy()

        for i, sweep_name in enumerate(sweep_names):
            curr_kwargs[sweep_name] = combos[i]

        sweep_list.append(curr_kwargs)

    return sweep_list
        
    

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

    output_metrics = {
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
        'compartment': [],
        'interior_threshold': [],
        'interior_smooth': [],
        'relevant_votes': [],
        'merge_radius': [],
        'maxima_threshold': [],
        'radius': [],
        'maxima_smooth': [],
        'postprocess_method': [],
        'small_objects_threshold': [],
        'fill_holes_threshold': [],
        'n_iter': [],
        'step_size': [],
        'maxima_algorithm': []
    }

    config = {
        'eval_info': 'data/segmentation/eval',
        'data_path': '/data/shared/tissuenet/tissuenet_v1.1_test.zarr',
        'model_path': 'data/model/20260213141132/saved_model_best_dict.pth'
    }

    # Whole cell, nuc
    sweep_hybrid = {
        'interior_threshold': [0.05, 0.2, 0.3],
        'merge_radius': [5, 7, 10, 15, 20]
    }

    sweep_classical = {
        'maxima_threshold': [0.05, 0.1, 0.15],
    }

    default_kwargs = {
            'small_objects_threshold': 15,
            'fill_holes_threshold': 15,
            'n_iter': 200,
            'step_size': 0.1,
            'maxima_threshold': 0.15,
            'maxima_smooth': 0,
            'interior_threshold': 0.3,
            'interior_smooth': 0.5,
            'radius': 2,
            'maxima_algorithm': 'h_maxima',
            'postprocess_method': 'hybrid',
            'relevant_votes': 2,
            'merge_radius': 10
        }

    sweep_list_hybrid = make_sweep(sweep_hybrid, default_kwargs, 'hybrid')
    sweep_list_classical = make_sweep(sweep_classical, default_kwargs, 'classical')
    all_sweeps = sweep_list_hybrid + sweep_list_classical
        
    z_test = zarr.open(f"{config['data_path']}")
    random_indices = random.sample(range(z_test['X'].shape[0]), 100)

    X_test = z_test['X'][random_indices]
    y_test = z_test['y'][random_indices]
    mpps = z_test['mpp'][random_indices]

    # Load model and application
    model = Mesmer(
        model_path = config['model_path'],
        device='cuda:2',
    )

    compartments = ['n','w']

    # evaluate the model
    for curr_sweep in tqdm(all_sweeps):

        preds = model.segment(X_test, mpps=mpps, postprocess_kwargs=curr_sweep)
        
        for c in range(X_test.shape[1]):
            output_metrics['compartment'].append(compartments[c])
            curr_metrics = evaluate(preds[:,c], y_test[:,c])

            for k, v in curr_metrics.items():
                output_metrics[k].append(v)
            for k, v in curr_sweep.items():
                output_metrics[k].append(v)

    processed_metrics = pd.DataFrame(output_metrics)
    processed_metrics.to_csv('parameter_sweep_metrics_finer2.csv')

if __name__ == "__main__":
    main()
