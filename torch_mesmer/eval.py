import glob
from pathlib import Path
from datetime import datetime
import zarr
import numpy as np

from torch_mesmer.mesmer import Mesmer
from torch_mesmer.metrics import Metrics

from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
import tqdm
import click

import pandas as pd

# Keys reported for each compartment. recall, precision and f1 are pooled over
# all images (micro-averaged), which is how deepcell's
# ``Metrics.summarize_object_metrics_df`` and the Mesmer paper report them.
SUMMARY_KEYS = [
    'recall', 'precision', 'f1',
    'n_true', 'n_pred', 'correct_detections',
    'gained_detections', 'missed_detections', 'split', 'merge', 'catastrophe',
    'jaccard', 'seg', 'seg_full',
]

SUMMARY_LABELS = {
    'recall': 'Recall (pooled)',
    'precision': 'Precision (pooled)',
    'f1': 'F1 (pooled)',
    'mean_per_image_recall': 'Recall (mean of per-image)',
    'mean_per_image_precision': 'Precision (mean of per-image)',
    'mean_per_image_f1': 'F1 (mean of per-image)',
    'n_true': 'True objects',
    'n_pred': 'Predicted objects',
    'correct_detections': 'Correct detections',
    'gained_detections': 'Gained detections',
    'missed_detections': 'Missed detections',
    'split': 'Split events',
    'merge': 'Merge events',
    'catastrophe': 'Many-to-many errors',
    'jaccard': 'Jaccard index (mean of per-image)',
    'seg': 'SEG, 1:1 matches only (mean of per-image)',
    'seg_full': 'SEG, standard (pooled over true objects)',
}


def evaluate(y_pred, y_true, progbar=False):
    """Score a batch of label images against the ground truth.

    Args:
        y_pred (numpy.array): predicted label images, shape (batch, x, y).
        y_true (numpy.array): ground-truth label images, same shape.
        progbar (bool): show a progress bar.

    Returns:
        tuple(pandas.DataFrame, dict): one row of object metrics per image, and
        the summary over the batch. In the summary, ``recall``, ``precision``
        and ``f1`` are **pooled**: correct detections, true objects and
        predicted objects are summed over all images before dividing, and
        ``f1`` is the harmonic mean of the pooled recall and precision. This is
        the convention of the Mesmer paper (F1 0.82, whole-cell). The
        ``mean_per_image_*`` keys give the mean of the per-image scores
        instead, for comparison only. Counts are summed over images.
    """
    m = Metrics("torch_mesmer")
    per_image = m.calc_object_stats(y_true, y_pred, progbar=progbar)
    pooled = m.summarize_object_metrics_df(per_image)

    summary = {k: pooled[k] for k in SUMMARY_KEYS}
    for k in ('recall', 'precision', 'f1'):
        summary[f'mean_per_image_{k}'] = float(per_image[k].mean())
    return per_image, summary


def pretty_print_summary(summaries: pd.DataFrame):
    """Print a summary table: one row per metric, one column per compartment."""
    order = [k for k in SUMMARY_LABELS if k in summaries.columns]
    # Format before transposing so counts print as whole numbers.
    fmt = lambda v: f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v)
    table = summaries[order].map(fmt).T.rename(index=SUMMARY_LABELS)

    print('-'*10 + ' Summary ' + '-'*10)
    print('Pooled = counts summed over all images before dividing (as in the Mesmer paper).')
    print()
    print(table.to_string())

    return None


def score_predictions(preds, y_true, compartments=('nuclear', 'whole-cell'), progbar=False):
    """Score multi-compartment predictions.

    Args:
        preds, y_true (numpy.array): label images, shape (batch, compartment, x, y).
        compartments: names of the compartments, in channel order.

    Returns:
        tuple(pandas.DataFrame, pandas.DataFrame): per-image metrics (with
        ``image`` and ``compartment`` columns), and one summary row per
        compartment.
    """
    per_image, summaries = [], {}
    for c, compartment in enumerate(compartments):
        df, summary = evaluate(preds[:, c], y_true[:, c], progbar=progbar)
        df.insert(0, 'compartment', compartment)
        df.insert(0, 'image', np.arange(len(df)))
        per_image.append(df)
        summaries[compartment] = summary
    summaries = pd.DataFrame.from_dict(summaries, orient='index')
    summaries.index.name = 'compartment'
    return pd.concat(per_image, ignore_index=True), summaries


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
            Can be `'cpu'` or `'cuda'`. If `'cuda'`, can also specify the specific GPU if multiple are available."""
)

@click.option(
    '--model-path',
    default=None,
    help="""Path to model.
            If unset, will select the latest available mesmer-*.pth from\n
            the canonical Deepcell models location ($HOME/.deepcell/models)"""
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
    """Segment every test image and score nuclear and whole-cell masks against the truth.

    Writes ``eval_results_<timestamp>.csv`` (one row per image) and
    ``eval_summary_<timestamp>.csv`` (pooled over images) to the current directory.

    Args:
        device: torch device to run the network on, e.g. ``'cuda:0'`` or ``'cpu'``.
        model_path: weights file (state_dict) to load.
        data_path: Zarr file with ``X``, ``y`` and ``meta`` (see ``preprocess.py``).

    Returns:
        ``(summaries, per_image)``, the two tables that were written to CSV.
    """
    z_test = zarr.open(data_path)

    X_test = z_test['X'][:]
    y_test = z_test['y'][:].astype(int)
    # NOTE: Zarr doesn't support structured arrays - must be converted to numpy
    # array explicitly before attempting to access fields
    mpps = z_test['meta'][:]["pixel_size"]

    # Default model path behavior: try to look up latest set of model weights
    if model_path is None:
        available_models = glob.glob(
            str(Path.home() / ".deepcell/models/torch-mesmer*.pth")
        )
        if not available_models:
            raise ValueError(
                "No model weights detected - use --model-path to specify path to model to evaluate."
            )
        # Select latest if multiple found
        model_path = sorted(available_models)[-1]
    print(f"Evaluating: {model_path}")


    # Load model and application
    model = Mesmer(
        model_path=model_path,
        device=device,
    )

    # The test set and the model output are nuclear first (heads 0-3 nuclear).
    compartments = ["nuclear", "whole-cell"]

    preds = np.zeros(y_test.shape, dtype=np.int32)
    for i in tqdm.tqdm(range(X_test.shape[0])):
        preds[i] = model.predict(X_test[i:i+1], image_mpp=mpps[i], compartment="both")[0]

    per_image, summaries = score_predictions(preds, y_test, compartments)

    pretty_print_summary(summaries)

    stamp = datetime.now().isoformat(timespec="seconds")
    per_image.to_csv(f'eval_results_{stamp}.csv', index=False)
    summaries.to_csv(f'eval_summary_{stamp}.csv')
    return summaries, per_image

if __name__ == "__main__":
    main()
