"""Grid search over Mesmer's post-processing settings.

Running the network is the slow part of ``Mesmer.predict``, and its output does
not depend on the post-processing settings. So the search has two steps:

1. ``cache`` runs the network once over a split and saves the four output maps
   that ``deep_watershed`` reads (inner distance and interior, for each
   compartment), at the model's resolution, exactly as ``Mesmer.predict`` would
   pass them on::

       python -m torch_mesmer.postprocess_sweep cache \\
           --model-path data/model/<run>/saved_model_best_dict.pth \\
           --data-path ~/.deepcell/tissuenet_v1-1/val.zarr --out data/postprocess_sweep/cache_val

2. ``sweep`` turns the saved maps into masks for every setting in a grid file,
   scores each mask against the ground truth, and writes one row per setting
   with pooled recall, precision and F1 (the Mesmer paper's convention) and
   the error-type counts::

       python -m torch_mesmer.postprocess_sweep sweep --cache data/postprocess_sweep/cache_val \\
           --grid grid.json --out results.csv [--subset 600] [--workers 8]

The masks are identical to ``Mesmer.predict``'s: the same ``deep_watershed``
steps, then ``resize_output`` back to the input size
(``tests/test_postprocess_sweep.py`` checks this). The speed-up comes from
reusing work shared between settings within an image: the smoothed maps, the
seeds (``h_maxima``) and the watershed itself.

Grid file format (JSON): a list of entries, each with a ``compartment``
(``"nuclear"`` or ``"whole-cell"``), a ``label``, and a ``grid`` mapping each
``deep_watershed`` keyword to a list of values. Every combination is tried.
Keywords left out take the value in ``BASE_KWARGS``.
"""

import hashlib
import itertools
import json
import os
import subprocess
import warnings
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
import pandas as pd
import scipy.ndimage as nd
from skimage.measure import label
from skimage.morphology import disk, footprint_rectangle, dilation, h_maxima, remove_small_objects
from skimage.segmentation import relabel_sequential, watershed

from torch_mesmer.postprocess_utils import MESMER_HEAD_INDICES, resize_output
from torch_mesmer.utils import fill_holes

COMPARTMENTS = ('nuclear', 'whole-cell')

# The cache keeps only the channels deep_watershed reads, in this order.
CACHE_CHANNELS = {
    'nuclear': {'maxima': 0, 'interior': 1},     # model channels 0 and 2
    'whole-cell': {'maxima': 2, 'interior': 3},  # model channels 4 and 6
}
MODEL_CHANNELS = [MESMER_HEAD_INDICES[c][k]
                  for c in COMPARTMENTS for k in ('maxima_index', 'interior_index')]

# Settings a grid entry doesn't mention.
BASE_KWARGS = {
    'maxima_threshold': 0.1,
    'maxima_smooth': 0,
    'interior_threshold': 0.2,
    'interior_smooth': 2,
    'radius': 2,
    'small_objects_threshold': 15,
    'fill_holes_threshold': 15,
    'pixel_expansion': 0,
}
SWEEP_KEYS = list(BASE_KWARGS)

# Per-image counts kept for every (image, setting); everything is pooled from these.
COUNT_FIELDS = ['n_true', 'n_pred', 'correct_detections', 'missed_detections',
                'gained_detections', 'split', 'merge', 'catastrophe',
                'missed_det_from_merge', 'gained_det_from_split',
                'true_det_in_catastrophe', 'pred_det_in_catastrophe']
PER_IMAGE_FIELDS = COUNT_FIELDS + ['f1']


# --------------------------------------------------------------------------- #
# Segmentation: the same steps as deep_watershed, sharing work across settings
# --------------------------------------------------------------------------- #

def segment_settings(maxima_map, interior_map, settings, original_hw):
    """Masks for several settings from one image's maps.

    Equivalent to running ``deep_watershed`` (``label_erosion=0``,
    ``maxima_algorithm='h_maxima'``) on the two maps for each setting, followed
    by ``resize_output(...).astype(int)`` as in ``Mesmer.predict``.

    Args:
        maxima_map, interior_map (numpy.array): 2D inner-distance and interior
            maps at the model's resolution.
        settings (list of dict): ``deep_watershed`` keywords; each must have
            every key in ``SWEEP_KEYS``.
        original_hw (tuple): (H, W) of the input image, before any rescaling.

    Returns:
        list of numpy.array: one 2D integer mask per setting, at ``original_hw``.
    """
    maxima_cache, marker_cache, interior_cache, ws_cache = {}, {}, {}, {}
    H, W = maxima_map.shape
    out = []
    for s in settings:
        ms = s['maxima_smooth']
        if ms not in maxima_cache:
            maxima_cache[ms] = nd.gaussian_filter(maxima_map, ms)
        mkey = (ms, s['radius'], s['maxima_threshold'])
        if mkey not in marker_cache:
            marker_cache[mkey] = label(h_maxima(image=maxima_cache[ms],
                                                h=s['maxima_threshold'],
                                                footprint=disk(s['radius'])))
        ikey = (s['interior_smooth'], s['pixel_expansion'])
        if ikey not in interior_cache:
            interior = nd.gaussian_filter(interior_map, s['interior_smooth'])
            pe = s['pixel_expansion']
            if pe:
                interior = dilation(interior, footprint=footprint_rectangle((pe * 2 + 1, pe * 2 + 1)))
            interior_cache[ikey] = interior
        interior = interior_cache[ikey]
        wkey = mkey + ikey + (s['interior_threshold'],)
        if wkey not in ws_cache:
            ws_cache[wkey] = watershed(-1 * interior, marker_cache[mkey],
                                       mask=interior > s['interior_threshold'],
                                       watershed_line=0)
        label_image = ws_cache[wkey]
        if s['small_objects_threshold']:
            label_image = remove_small_objects(label_image, min_size=s['small_objects_threshold'])
        if s['fill_holes_threshold'] > 0:
            label_image = fill_holes(label_image, size=s['fill_holes_threshold'])
        label_image, _, _ = relabel_sequential(label_image)

        # deep_watershed returns a float array; resize_output then works on that.
        as_float = np.zeros((1, 1, H, W))
        as_float[0, 0] = label_image
        mask = resize_output(as_float, (1, 1) + tuple(original_hw)).astype(int)
        out.append(mask[0, 0])
    return out


def score_mask(y_true, y_pred):
    """Per-image object metrics, as ``Metrics.calc_object_stats`` computes them."""
    from torch_mesmer.metrics import ObjectMetrics
    o = ObjectMetrics(relabel_sequential(y_true)[0], relabel_sequential(y_pred)[0]).to_dict()
    return [o[k] for k in PER_IMAGE_FIELDS]


def pool_counts(counts):
    """Pooled metrics from summed per-image counts (as ``summarize_object_metrics_df``).

    Args:
        counts (numpy.array): shape (n_images, n_fields), columns ``PER_IMAGE_FIELDS``.
    """
    s = dict(zip(COUNT_FIELDS, counts[:, :len(COUNT_FIELDS)].sum(axis=0).astype(int)))
    recall = s['correct_detections'] / s['n_true'] if s['n_true'] else np.nan
    precision = s['correct_detections'] / s['n_pred'] if s['n_pred'] else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    return {'recall': recall, 'precision': precision, 'f1': f1,
            'mean_per_image_f1': float(np.nanmean(counts[:, -1])), **s}


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

def _sha256(path):
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    """Current git commit hash, or ``'unknown'`` if it cannot be read.

    ``-dirty`` is added when files under ``torch_mesmer/`` have uncommitted
    changes. Changes elsewhere in the repository are not counted.
    """
    try:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        dirty = subprocess.call(['git', 'diff', '--quiet', 'HEAD', '--', 'torch_mesmer'], cwd=root)
        return commit + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def build_cache(model_path, data_path, out_dir, device, batch_size=4):
    """Run the network once over a split and save the maps deep_watershed reads."""
    import torch
    import zarr
    from tqdm import tqdm
    from torch_mesmer.mesmer import Mesmer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z = zarr.open(str(data_path), mode='r')
    meta = pd.DataFrame(z['meta'][:])  # zarr cannot index a structured array directly: read it into numpy first
    mpps = meta['pixel_size'].to_numpy()
    n = z['X'].shape[0]

    app = Mesmer(model_path=model_path, device=device)
    groups, rows = {}, []
    for i in tqdm(range(n)):
        x = z['X'][i:i + 1]
        mpp = None if np.isnan(mpps[i]) else float(mpps[i])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            _, transforms = app.predict(x, batch_size=batch_size, image_mpp=mpp,
                                        compartment='nuclear', return_transforms=True)
        maps = np.asarray(transforms[0, MODEL_CHANNELS], dtype=np.float32)
        key = f'{maps.shape[1]}x{maps.shape[2]}'
        groups.setdefault(key, []).append(maps)
        rows.append({'image': i, 'group': key, 'row': len(groups[key]) - 1,
                     'orig_h': x.shape[2], 'orig_w': x.shape[3]})

    for key, maps in groups.items():
        np.save(out_dir / f'maps_{key}.npy', np.stack(maps))
    index = pd.DataFrame(rows)
    index = pd.concat([index, meta.reset_index(drop=True)], axis=1)
    index.to_csv(out_dir / 'index.csv', index=False)
    info = {
        'model_path': str(Path(model_path).resolve()),
        'model_sha256': _sha256(model_path),
        'data_path': str(Path(data_path).resolve()),
        'n_images': n,
        'channels': {c: [MESMER_HEAD_INDICES[c]['maxima_index'], MESMER_HEAD_INDICES[c]['interior_index']]
                     for c in COMPARTMENTS},
        'git_commit': _git_commit(),
        'created': datetime.now().isoformat(timespec='seconds'),
        'torch_version': torch.__version__,
    }
    (out_dir / 'cache_info.json').write_text(json.dumps(info, indent=2))
    return info


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def expand_grid(entries):
    """List of (compartment, label, kwargs) from grid-file entries, without duplicates."""
    out, seen = [], set()
    for e in entries:
        compartment = e['compartment']
        if compartment not in COMPARTMENTS:
            raise ValueError(f'Unknown compartment {compartment!r}')
        unknown = set(e['grid']) - set(SWEEP_KEYS)
        if unknown:
            raise ValueError(f'Unknown keywords in grid {e["label"]!r}: {sorted(unknown)}')
        keys = list(e['grid'])
        for combo in itertools.product(*(e['grid'][k] for k in keys)):
            kw = {**BASE_KWARGS, **dict(zip(keys, combo))}
            sig = (compartment, tuple(kw[k] for k in SWEEP_KEYS))
            if sig in seen:
                # Keep the first label, but remember the other one too.
                for row in out:
                    if (row[0], tuple(row[2][k] for k in SWEEP_KEYS)) == sig and e['label'] not in row[1]:
                        row[1].append(e['label'])
                continue
            seen.add(sig)
            out.append((compartment, [e['label']], kw))
    return out


_W = {}  # per-worker state


def _init_worker(cache_dir, data_path, settings_by_comp):
    """Load the cached maps and labels once in each worker process.

    The state goes into the module-level ``_W`` dict, which ``_work`` reads.
    The ``OMP_NUM_THREADS`` default only helps if it is set before numpy is
    imported, so in practice it has no effect here.
    """
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    warnings.simplefilter('ignore')
    import zarr
    cache_dir = Path(cache_dir)
    _W['index'] = pd.read_csv(cache_dir / 'index.csv')
    _W['maps'] = {p.stem[len('maps_'):]: np.load(p, mmap_mode='r') for p in cache_dir.glob('maps_*.npy')}
    _W['y'] = zarr.open(str(data_path), mode='r')['y']
    _W['settings'] = settings_by_comp


def _work(image):
    """Score every setting of every compartment on one cached image.

    Args:
        image: index of the image in the split.

    Returns:
        ``(image, scores)``: ``scores`` maps each compartment name to a float64
        array of shape ``(n_settings, n_fields)``, with one row per setting (in
        the order given to ``_init_worker``) and one column per entry of
        ``PER_IMAGE_FIELDS``. Compartments without settings are left out.
    """
    row = _W['index'].iloc[image]
    maps = np.asarray(_W['maps'][row['group']][row['row']])
    y = np.asarray(_W['y'][image]).astype(int)
    result = {}
    for c, compartment in enumerate(COMPARTMENTS):
        settings = _W['settings'].get(compartment)
        if not settings:
            continue
        ch = CACHE_CHANNELS[compartment]
        masks = segment_settings(maps[ch['maxima']], maps[ch['interior']], settings,
                                 (row['orig_h'], row['orig_w']))
        result[compartment] = np.array([score_mask(y[c], m) for m in masks], dtype=np.float64)
    return image, result


def run_sweep(cache_dir, grid_entries, out_csv, subset=None, seed=0, workers=8, images=None):
    """Score every grid setting over the cached images; write a CSV and per-image counts."""
    from tqdm import tqdm

    cache_dir = Path(cache_dir)
    info = json.loads((cache_dir / 'cache_info.json').read_text())
    index = pd.read_csv(cache_dir / 'index.csv')
    if images is None:
        images = np.arange(len(index))
        if subset:
            images = np.sort(np.random.default_rng(seed).choice(images, size=subset, replace=False))
    images = np.asarray(images)

    settings = expand_grid(grid_entries)
    by_comp = {c: [s for s in settings if s[0] == c] for c in COMPARTMENTS}
    kwargs_by_comp = {c: [s[2] for s in v] for c, v in by_comp.items()}
    per_image = {c: np.zeros((len(images), len(v), len(PER_IMAGE_FIELDS))) for c, v in by_comp.items()}
    pos = {img: k for k, img in enumerate(images)}
    print(f'{len(images)} images; settings: ' +
          ', '.join(f'{c} {len(v)}' for c, v in by_comp.items()) + f'; {workers} workers')

    with Pool(workers, initializer=_init_worker,
              initargs=(str(cache_dir), info['data_path'], kwargs_by_comp)) as pool:
        for image, result in tqdm(pool.imap_unordered(_work, images.tolist(), chunksize=2),
                                  total=len(images)):
            for c, arr in result.items():
                per_image[c][pos[image]] = arr

    rows = []
    for c in COMPARTMENTS:
        for k, (_, labels, kw) in enumerate(by_comp[c]):
            rows.append({'compartment': c, 'label': '+'.join(labels), **kw,
                         **pool_counts(per_image[c][:, k])})
    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    np.savez_compressed(out_csv.with_suffix('.per_image.npz'), images=images,
                        fields=np.array(PER_IMAGE_FIELDS),
                        **{c.replace('-', '_'): v for c, v in per_image.items()})
    out_csv.with_suffix('.json').write_text(json.dumps({
        'cache': str(cache_dir.resolve()), 'cache_info': info, 'grid': grid_entries,
        'n_images': int(len(images)), 'subset': subset, 'seed': seed,
        'git_commit': _git_commit(), 'created': datetime.now().isoformat(timespec='seconds'),
    }, indent=2))
    return df


@click.group()
def cli():
    pass


@cli.command()
@click.option('--model-path', required=True, help='Weights file (.pth).')
@click.option('--data-path', default=str(Path.home() / '.deepcell/tissuenet_v1-1/val.zarr'),
              show_default=True, help='Zarr split to run on. Tune on val, not test.')
@click.option('--out', 'out_dir', required=True, help='Output folder for the cached maps.')
@click.option('--device', default='cuda', show_default=True)
@click.option('--threads', default=4, show_default=True, help='CPU threads for torch (shared server).')
def cache(model_path, data_path, out_dir, device, threads):
    """Run the network once and save the maps deep_watershed reads."""
    import torch
    torch.set_num_threads(threads)
    info = build_cache(model_path, data_path, out_dir, device)
    print(json.dumps(info, indent=2))


@cli.command()
@click.option('--cache', 'cache_dir', required=True, help='Folder written by `cache`.')
@click.option('--grid', 'grid_path', required=True, help='Grid file (JSON).')
@click.option('--out', 'out_csv', required=True, help='Output CSV.')
@click.option('--subset', default=None, type=int, help='Score a random subset of this many images.')
@click.option('--seed', default=0, show_default=True)
@click.option('--workers', default=8, show_default=True, help='Worker processes (shared server).')
def sweep(cache_dir, grid_path, out_csv, subset, seed, workers):
    """Score every setting in the grid file over the cached images."""
    entries = json.loads(Path(grid_path).read_text())
    df = run_sweep(cache_dir, entries, out_csv, subset=subset, seed=seed, workers=workers)
    cols = ['label', 'recall', 'precision', 'f1', 'merge', 'split']
    for c in COMPARTMENTS:
        top = df[df.compartment == c].sort_values('f1', ascending=False).head(10)
        if len(top):
            print(f'\n{c}: top 10 by pooled F1')
            print(top[cols + SWEEP_KEYS].to_string(index=False))


if __name__ == '__main__':
    cli()
