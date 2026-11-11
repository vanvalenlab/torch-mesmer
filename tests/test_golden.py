"""Golden-output regression test for ``Mesmer.predict`` with the released weights.

Later fixes to inference and API bugs must not change any mask that ``predict``
already produced. ``tests/data/golden_masks.npz`` holds masks made by ``predict``
*before* those fixes, plus the exact inputs and the SHA-256 of the weights file.
This test re-runs every case and requires identical masks.

The cases cover each path through ``predict`` that worked before those fixes: the
three compartments, the resize path (``image_mpp`` != 0.5), the tiling path
(an image larger than 256) and the padding path (an image smaller than 256).

It takes about 3 minutes on 4 CPU threads, so it is deselected by default.
Run it with ``.venv/bin/pytest -m golden`` after any change to inference code.

Regenerate (only when the weights change on purpose, e.g. after a retrain) with::

    OMP_NUM_THREADS=4 .venv/bin/python -m tests.test_golden [path/to/weights.pth]
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent / 'data' / 'golden_masks.npz'
DEFAULT_WEIGHTS = Path.home() / '.deepcell' / 'models' / 'torch-mesmer_2026-06-30.pth'
TEST_ZARR = Path.home() / '.deepcell' / 'tissuenet_v1-1' / 'test.zarr'

# Test-split images spread across the 1320-image split (different tissues/platforms).
IMAGE_INDICES = [0, 300, 600, 900, 1200]

# name -> (which input, predict kwargs). Inputs are built in ``make_inputs``.
CASES = {
    'nuclear_mpp0.5': ('tiles', dict(image_mpp=0.5, compartment='nuclear')),
    'whole-cell_mpp0.5': ('tiles', dict(image_mpp=0.5, compartment='whole-cell')),
    'both_mpp0.5': ('tiles', dict(image_mpp=0.5, compartment='both')),
    'whole-cell_mpp0.65': ('tiles', dict(image_mpp=0.65, compartment='whole-cell')),
    'both_mpp0.4': ('tiles', dict(image_mpp=0.4, compartment='both')),
    'both_mosaic512': ('mosaic', dict(image_mpp=0.5, compartment='both')),
    'both_crop200': ('crop', dict(image_mpp=0.5, compartment='both')),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def make_inputs(tiles):
    """``tiles``: (5, 2, 256, 256) test images. Returns the three input arrays."""
    top = np.concatenate([tiles[0], tiles[1]], axis=-1)
    bottom = np.concatenate([tiles[2], tiles[3]], axis=-1)
    mosaic = np.concatenate([top, bottom], axis=-2)[None]  # (1, 2, 512, 512): tiling path
    crop = tiles[4:5, :, 28:228, 28:228]                    # (1, 2, 200, 200): padding path
    return {'tiles': tiles, 'mosaic': np.ascontiguousarray(mosaic), 'crop': np.ascontiguousarray(crop)}


# The post-processing defaults the golden masks were made with (those in
# ``mesmer.py`` today). Passing them explicitly keeps this test about the
# inference code, independent of later changes to the defaults.
POSTPROCESS_NUCLEAR = {'maxima_threshold': 0.1, 'maxima_smooth': 0, 'interior_threshold': 0.075,
                       'interior_smooth': 1, 'small_objects_threshold': 15, 'fill_holes_threshold': 15,
                       'radius': 2, 'pixel_expansion': 0}
POSTPROCESS_WHOLE_CELL = {**POSTPROCESS_NUCLEAR, 'maxima_smooth': 1}


def run_cases(app, inputs):
    return {name: app.predict(inputs[which], postprocess_kwargs_nuclear=POSTPROCESS_NUCLEAR,
                              postprocess_kwargs_whole_cell=POSTPROCESS_WHOLE_CELL, **kwargs)
            for name, (which, kwargs) in CASES.items()}


@pytest.mark.golden
def test_predict_matches_golden_masks():
    if not GOLDEN.exists():
        pytest.skip(f'{GOLDEN} not found')
    golden = np.load(GOLDEN)
    weights = Path(str(golden['weights_path']))
    if not weights.exists():
        pytest.skip(f'released weights {weights} not found')
    if sha256(weights) != str(golden['weights_sha256']):
        pytest.skip(f'{weights} is not the file the golden masks were made with; regenerate them')

    from torch_mesmer.mesmer import Mesmer
    app = Mesmer(model_path=str(weights), device='cpu')
    inputs = make_inputs(golden['input_tiles'])

    for name, mask in run_cases(app, inputs).items():
        expected = golden[f'mask_{name}']
        assert mask.shape == expected.shape, name
        n_diff = int((mask != expected).sum())
        assert n_diff == 0, f'{name}: {n_diff} pixels differ from the golden mask'


def main(weights=DEFAULT_WEIGHTS):
    import torch
    import zarr

    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
    from torch_mesmer.mesmer import Mesmer

    tiles = np.asarray(zarr.open(str(TEST_ZARR), mode='r')['X'].get_orthogonal_selection((IMAGE_INDICES,)))
    tiles = tiles.astype(np.float32)
    app = Mesmer(model_path=str(weights), device='cpu')
    masks = run_cases(app, make_inputs(tiles))

    GOLDEN.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        GOLDEN,
        input_tiles=tiles,
        image_indices=np.array(IMAGE_INDICES),
        weights_path=str(weights),
        weights_sha256=sha256(weights),
        **{f'mask_{name}': m.astype(np.int32) for name, m in masks.items()},
    )
    for name, m in masks.items():
        print(f'{name:22s} shape={m.shape} n_objects={[int(np.unique(c).size - 1) for c in m[:, 0]]}')
    print(f'wrote {GOLDEN} ({GOLDEN.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main(*sys.argv[1:])
