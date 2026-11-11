"""Reference (deepcell) implementations for the parity tests.

The real ``deepcell_toolbox`` package needs a compiled Cython extension
(``compute_overlap.pyx``) and sklearn, and ``deepcell-tf`` needs TensorFlow.
None of those are installed in ``.venv``. Instead of importing the packages
normally, this module:

* loads the toolbox's ``utils.py``, ``processing.py``, ``deep_watershed.py`` and
  ``metrics.py`` straight from their source files in the local
  ``deepcell-toolbox/`` checkout;
* supplies pure-Python stand-ins for the pieces that can't be imported here
  (the Cython overlap functions, ``sklearn.metrics.confusion_matrix`` and the
  three ``tensorflow.keras.backend`` helpers that ``transform_utils.py`` uses);
* transcribes the handful of small functions that live inside TensorFlow-only
  modules (``deepcell/applications/*.py``), with a comment naming the source.

The stand-ins are put into ``sys.modules`` only while the reference files are
being executed and are removed afterwards, so they never leak into the port or
into other tests.

Usage::

    from tests import _reference
    ref = _reference.load()          # raises ReferenceUnavailable if checkouts are missing
    ref.metrics.ObjectMetrics(...)
"""

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLBOX_DIR = REPO_ROOT / 'deepcell-toolbox' / 'deepcell_toolbox'
DEEPCELL_TF_DIR = REPO_ROOT / 'deepcell-tf' / 'deepcell'


class ReferenceUnavailable(RuntimeError):
    """The local deepcell-toolbox / deepcell-tf checkouts are missing."""


# ---------------------------------------------------------------------------
# Stand-ins for compiled / heavy dependencies
# ---------------------------------------------------------------------------

def compute_overlap(boxes, query_boxes):
    """Pure-Python transcription of ``compute_overlap`` from
    ``deepcell-toolbox/deepcell_toolbox/compute_overlap.pyx:13-52``.

    Boxes are ``(N, 4)`` arrays of ``[x1, y1, x2, y2]`` (skimage bbox order).
    The loop structure and the ``+ 1`` inclusive-pixel convention are kept
    exactly as in the Cython source.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    query_boxes = np.asarray(query_boxes, dtype=np.float64)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def compute_overlap_3D(boxes, query_boxes):
    """Pure-Python transcription of ``compute_overlap_3D`` from
    ``deepcell-toolbox/deepcell_toolbox/compute_overlap.pyx:55-103``.

    Boxes are ``(N, 6)`` arrays of ``[z1, x1, y1, z2, x2, y2]``.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    query_boxes = np.asarray(query_boxes, dtype=np.float64)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_volume = (
            (query_boxes[k, 3] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 4] - query_boxes[k, 1] + 1) *
            (query_boxes[k, 5] - query_boxes[k, 2] + 1)
        )
        for n in range(N):
            id_ = (
                min(boxes[n, 3], query_boxes[k, 3]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if id_ > 0:
                iw = (
                    min(boxes[n, 4], query_boxes[k, 4]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if iw > 0:
                    ih = (
                        min(boxes[n, 5], query_boxes[k, 5]) -
                        max(boxes[n, 2], query_boxes[k, 2]) + 1
                    )
                    if ih > 0:
                        ua = np.float64(
                            (boxes[n, 3] - boxes[n, 0] + 1) *
                            (boxes[n, 4] - boxes[n, 1] + 1) *
                            (boxes[n, 5] - boxes[n, 2] + 1) +
                            box_volume - iw * ih * id_
                        )
                        overlaps[n, k] = iw * ih * id_ / ua
    return overlaps


def confusion_matrix(y_true, y_pred):
    """Minimal stand-in for ``sklearn.metrics.confusion_matrix`` with default
    arguments: rows are true classes, columns are predicted classes, and the
    class list is the sorted union of the values seen in either input."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = np.union1d(y_true, y_pred)
    index = {v: i for i, v in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm


def _fake_keras_backend():
    """The three ``tensorflow.keras.backend`` helpers used by
    ``deepcell-tf/deepcell/utils/transform_utils.py``, with Keras defaults."""
    K = types.ModuleType('tensorflow.keras.backend')
    K.image_data_format = lambda: 'channels_last'
    K.floatx = lambda: 'float32'
    K.epsilon = lambda: 1e-7
    return K


# ---------------------------------------------------------------------------
# Transcriptions of small functions from TensorFlow-only modules
# ---------------------------------------------------------------------------

def _make_application_functions(toolbox_utils, toolbox_processing, toolbox_dw):
    """Build reference versions of the Mesmer application helpers.

    These live in ``deepcell-tf/deepcell/applications/{application,mesmer}.py``,
    which import TensorFlow at module level, so they are transcribed here. All
    are channels-last, like the originals. Only the ``self.`` attributes were
    turned into arguments.
    """
    resize = toolbox_utils.resize
    tile_image = toolbox_utils.tile_image
    untile_image = toolbox_utils.untile_image

    # deepcell-tf/deepcell/applications/mesmer.py:42-66
    def mesmer_preprocess(image, **kwargs):
        if len(image.shape) != 4:
            raise ValueError(f"Image data must be 4D, got image of shape {image.shape}")

        output = np.copy(image)
        threshold = kwargs.get('threshold', True)
        if threshold:
            percentile = kwargs.get('percentile', 99.9)
            output = toolbox_processing.percentile_threshold(image=output, percentile=percentile)

        normalize = kwargs.get('normalize', True)
        if normalize:
            kernel_size = kwargs.get('kernel_size', 128)
            output = toolbox_processing.histogram_normalization(image=output, kernel_size=kernel_size)

        return output

    # deepcell-tf/deepcell/applications/mesmer.py:69-91
    def format_output_mesmer(output_list):
        expected_length = 4
        if len(output_list) != expected_length:
            raise ValueError('output_list was length {}, expecting length {}'.format(
                len(output_list), expected_length))

        return {
            'whole-cell': [output_list[0], output_list[1][..., 1:2]],
            'nuclear': [output_list[2], output_list[3][..., 1:2]],
        }

    # deepcell-tf/deepcell/applications/mesmer.py:94-145
    def mesmer_postprocess(model_output, compartment='whole-cell',
                           whole_cell_kwargs=None, nuclear_kwargs=None):
        valid_compartments = ['whole-cell', 'nuclear', 'both']
        whole_cell_kwargs = whole_cell_kwargs or {}
        nuclear_kwargs = nuclear_kwargs or {}
        if compartment not in valid_compartments:
            raise ValueError(f'Invalid compartment supplied: {compartment}. '
                             f'Must be one of {valid_compartments}')
        if compartment == 'whole-cell':
            return toolbox_dw.deep_watershed(model_output['whole-cell'], **whole_cell_kwargs)
        if compartment == 'nuclear':
            return toolbox_dw.deep_watershed(model_output['nuclear'], **nuclear_kwargs)
        label_images_cell = toolbox_dw.deep_watershed(model_output['whole-cell'], **whole_cell_kwargs)
        label_images_nucleus = toolbox_dw.deep_watershed(model_output['nuclear'], **nuclear_kwargs)
        return np.concatenate([label_images_cell, label_images_nucleus], axis=-1)

    # deepcell-tf/deepcell/applications/application.py:79-100 (Application._resize_input)
    def resize_input(image, image_mpp, model_mpp):
        if image_mpp not in {None, model_mpp}:
            shape = image.shape
            scale_factor = image_mpp / model_mpp
            new_shape = (int(shape[1] * scale_factor),
                         int(shape[2] * scale_factor))
            image = resize(image, new_shape, data_format='channels_last')
        return image

    # deepcell-tf/deepcell/applications/application.py:126-170 (Application._tile_input)
    def tile_input(image, model_image_shape, pad_mode='constant'):
        if len(image.shape) != 4:
            raise ValueError('deepcell_toolbox.tile_image only supports 4d images.'
                             f'Image submitted for predict has {len(image.shape)} dimensions')

        x_diff = image.shape[1] - model_image_shape[0]
        y_diff = image.shape[2] - model_image_shape[1]

        if x_diff < 0 or y_diff < 0:
            x_diff, y_diff = abs(x_diff), abs(y_diff)
            x_pad = (x_diff // 2, x_diff // 2 + 1) if x_diff % 2 else (x_diff // 2, x_diff // 2)
            y_pad = (y_diff // 2, y_diff // 2 + 1) if y_diff % 2 else (y_diff // 2, y_diff // 2)

            tiles = np.pad(image, [(0, 0), x_pad, y_pad, (0, 0)], 'reflect')
            tiles_info = {'padding': True, 'x_pad': x_pad, 'y_pad': y_pad}
        else:
            tiles, tiles_info = tile_image(image, model_input_shape=model_image_shape,
                                           stride_ratio=0.75, pad_mode=pad_mode)
        return tiles, tiles_info

    # deepcell-tf/deepcell/applications/application.py:201-233 (Application._untile_output)
    def untile_output(output_tiles, tiles_info, model_image_shape):
        if tiles_info.get('padding', False):
            def _process(im, tiles_info):
                ((xl, xh), (yl, yh)) = tiles_info['x_pad'], tiles_info['y_pad']
                xh = -xh if xh != 0 else None
                yh = -yh if yh != 0 else None
                return im[:, xl:xh, yl:yh, :]
        else:
            def _process(im, tiles_info):
                return untile_image(im, tiles_info, model_input_shape=model_image_shape)

        if isinstance(output_tiles, list):
            return [_process(o, tiles_info) for o in output_tiles]
        return _process(output_tiles, tiles_info)

    # deepcell-tf/deepcell/applications/application.py:252-288 (Application._resize_output)
    def resize_output(image, original_shape):
        if not isinstance(image, list):
            image = [image]
        for i in range(len(image)):
            img = image[i]
            if len(img.shape) == 4:
                same = img.shape[1:-1] == original_shape[1:-1]
            elif len(img.shape) == 3:
                same = img.shape[1:] == original_shape[1:-1]
            else:
                same = img.shape == original_shape[1:-1]
            if not same:
                new_shape = original_shape[1:-1]
                img = resize(img, new_shape, data_format='channels_last', labeled_image=True)
            image[i] = img
        if len(image) == 1:
            image = image[0]
        return image

    return types.SimpleNamespace(
        mesmer_preprocess=mesmer_preprocess,
        format_output_mesmer=format_output_mesmer,
        mesmer_postprocess=mesmer_postprocess,
        resize_input=resize_input,
        tile_input=tile_input,
        untile_output=untile_output,
        resize_output=resize_output,
    )


# ---------------------------------------------------------------------------
# Loading machinery
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _temporary_modules(**modules):
    """Put ``modules`` into ``sys.modules`` for the duration of the block, then
    restore whatever was there before (including "nothing")."""
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _exec_file(module_name, path):
    """Execute a source file as a module named ``module_name`` and return it.
    The module is also registered in ``sys.modules`` while executing so that
    later reference files can import it by name; ``_temporary_modules`` takes
    care of removing it again."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_CACHE = None


def load():
    """Load every reference module once and return them in a namespace.

    Attributes of the returned namespace:

    * ``utils``, ``processing``, ``deep_watershed``, ``metrics``: the toolbox
      modules (``deepcell-toolbox/deepcell_toolbox/*.py``).
    * ``transform_utils``: ``deepcell-tf/deepcell/utils/transform_utils.py``.
    * ``compute_overlap``, ``compute_overlap_3D``, ``confusion_matrix``: the
      stand-ins defined above.
    * ``app``: transcribed Mesmer application helpers (channels-last).
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    needed = [TOOLBOX_DIR / f for f in ('utils.py', 'processing.py', 'deep_watershed.py', 'metrics.py')]
    needed.append(DEEPCELL_TF_DIR / 'utils' / 'transform_utils.py')
    absent = [str(p.relative_to(REPO_ROOT)) for p in needed if not p.is_file()]
    if absent:
        raise ReferenceUnavailable(
            'Reference checkouts not found (expected deepcell-toolbox/ and deepcell-tf/ '
            'at the repo root). Missing: ' + ', '.join(absent))

    # A bare package object standing in for ``deepcell_toolbox/__init__.py``,
    # which would otherwise try to import the compiled extension.
    pkg = types.ModuleType('deepcell_toolbox')
    pkg.__path__ = [str(TOOLBOX_DIR)]

    overlap_mod = types.ModuleType('deepcell_toolbox.compute_overlap')
    overlap_mod.compute_overlap = compute_overlap
    overlap_mod.compute_overlap_3D = compute_overlap_3D

    sklearn_mod = types.ModuleType('sklearn')
    sklearn_metrics_mod = types.ModuleType('sklearn.metrics')
    sklearn_metrics_mod.confusion_matrix = confusion_matrix
    sklearn_mod.metrics = sklearn_metrics_mod

    K = _fake_keras_backend()
    tf_mod = types.ModuleType('tensorflow')
    tf_keras_mod = types.ModuleType('tensorflow.keras')
    tf_keras_mod.backend = K
    tf_mod.keras = tf_keras_mod

    stand_ins = {
        'deepcell_toolbox': pkg,
        'deepcell_toolbox.compute_overlap': overlap_mod,
        # Placeholders so _temporary_modules also removes these after loading.
        'deepcell_toolbox.utils': None,
        'deepcell_toolbox.processing': None,
        'deepcell_toolbox.deep_watershed': None,
        'deepcell_toolbox.metrics': None,
        'sklearn': sklearn_mod,
        'sklearn.metrics': sklearn_metrics_mod,
        'tensorflow': tf_mod,
        'tensorflow.keras': tf_keras_mod,
        'tensorflow.keras.backend': K,
        '_ref_transform_utils': None,
    }

    with _temporary_modules(**stand_ins):
        utils = _exec_file('deepcell_toolbox.utils', TOOLBOX_DIR / 'utils.py')
        processing = _exec_file('deepcell_toolbox.processing', TOOLBOX_DIR / 'processing.py')

        # Mirror what deepcell_toolbox/__init__.py exposes at package level.
        pkg.utils = utils
        pkg.processing = processing
        pkg.erode_edges = utils.erode_edges
        pkg.compute_overlap = compute_overlap
        pkg.compute_overlap_3D = compute_overlap_3D
        for name in ('normalize', 'phase_preprocess', 'histogram_normalization',
                     'mibi', 'watershed', 'pixelwise'):
            if hasattr(processing, name):
                setattr(pkg, name, getattr(processing, name))

        deep_watershed = _exec_file('deepcell_toolbox.deep_watershed', TOOLBOX_DIR / 'deep_watershed.py')
        metrics = _exec_file('deepcell_toolbox.metrics', TOOLBOX_DIR / 'metrics.py')
        transform_utils = _exec_file('_ref_transform_utils', DEEPCELL_TF_DIR / 'utils' / 'transform_utils.py')

    _CACHE = types.SimpleNamespace(
        utils=utils,
        processing=processing,
        deep_watershed=deep_watershed,
        metrics=metrics,
        transform_utils=transform_utils,
        compute_overlap=compute_overlap,
        compute_overlap_3D=compute_overlap_3D,
        confusion_matrix=confusion_matrix,
        app=_make_application_functions(utils, processing, deep_watershed),
    )
    return _CACHE
