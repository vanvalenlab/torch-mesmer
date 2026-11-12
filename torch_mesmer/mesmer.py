from pathlib import Path

import numpy as np
import torch

from torch_mesmer.model import PanopticNet, LEGACY_ARCH_VERSION, checkpoint_arch_version

from torch_mesmer.postprocess_utils import resize_input, resize_output, mesmer_postprocess, mesmer_preprocess, untile_output, tile_input

# The Mesmer network's output channels: 4 per compartment, nuclear first
# (see ``PanopticNet`` for what each channel holds).
_MESMER_OUTPUT_CHANNELS = 8

# Post-processing defaults for ``Mesmer.predict``, one set per compartment.
# They differ from deepcell's; ``torch_mesmer/postprocess_sweep.py`` is the
# tool for re-tuning them, which should be done after any retrain. This is the
# single source of truth for them. Head indices are not listed here:
# ``mesmer_postprocess`` takes them from ``postprocess_utils.MESMER_HEAD_INDICES``.
DEFAULT_POSTPROCESS_KWARGS_NUCLEAR = {
    'maxima_threshold': 0.1,
    'maxima_smooth': 0,
    'interior_threshold': 0.075,
    'interior_smooth': 1,
    'small_objects_threshold': 15,
    'fill_holes_threshold': 15,
    'radius': 2,
    'pixel_expansion': 0,
}

DEFAULT_POSTPROCESS_KWARGS_WHOLE_CELL = {
    'maxima_threshold': 0.1,
    'maxima_smooth': 1,
    'interior_threshold': 0.075,
    'interior_smooth': 1,
    'small_objects_threshold': 15,
    'fill_holes_threshold': 15,
    'radius': 2,
    'pixel_expansion': 0,
}


# The released weights that ``Mesmer()`` uses when no ``model_path`` is given:
# a Hugging Face repository and the file in it. They are saved to
# ``~/.deepcell/models`` the first time.
_RELEASED_WEIGHTS_REPO = 'vanvalenlab/torch-mesmer'
_RELEASED_WEIGHTS_FILE = 'torch-mesmer_2026-06-30.pth'


def _released_weights_path():
    """Download the released weights from Hugging Face if they are not in
    ``~/.deepcell/models`` yet, and return their local path (str)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=_RELEASED_WEIGHTS_REPO,
        filename=_RELEASED_WEIGHTS_FILE,
        local_dir=Path.home() / '.deepcell' / 'models',
    )


class Mesmer():
    """The Mesmer cell segmentation pipeline for multiplexed images.

    The input to this pipeline is a multiplexed image with two channels: the
    first channel should represent a nuclear marker and the second channel a cell
    membrane or cytosol marker. It is then possible to perform nuclear segmentation
    with ``compartment="nuclear"``, whole-cell segmentation with
    ``compartment="whole-cell"``, or both nuclear and cell segmentation
    simultaneously with ``compartment="both"``. Everything is nuclear first: the
    input channels, the network's output channels and the masks.

    `predict` runs the whole pipeline: rescale to the model's resolution,
    normalise, tile, run the network, stitch the tiles together, and turn the
    network output into label masks. See the `predict` docstring for details.
    """
    def __init__(
        self, model_path=None, device=None, n_semantic_classes=[1, 3, 1, 3]
    ):
        """
        Instantiate an instance of the Mesmer cell segmentation pipeline.

        The returned instance is designed for segmenting whole-slide images.

        Weights with no ``arch_version`` entry (the 2026-06-30 release) are loaded
        into the legacy network, ``PanopticNet(legacy=True)``.

        Parameters
        ----------
        model_path : str or pathlib.Path, default=None
            The path to the trained model (i.e. a ``.pth`` file).
            If not specified (the default), the released weights are used, after
            downloading them from huggingface into ``~/.deepcell/models`` if needed.

            .. note::
               Internet access and a valid ``HF_TOKEN`` is required to download
               the latest weights. See :doc:`/model_access` for details.

        device : str, default=None
            A `torch.device` compatible specifier indicating the hardware to be
            used for inference, e.g. ``"cuda"``, ``"mps"``, or ``"cpu"``.
            If not specified (the default), Mesmer will use a cuda or mps-compatible
            backend if it is detected, falling back to ``"cpu"`` if not.

        n_semantic_classes : list of int, default=[1, 3, 1, 3]
            Output channels of each head of the network; ``[1, 3, 1, 3]`` for Mesmer.
            `predict` assumes Mesmer's 8 output channels, so other values (such as
            ``[1, 3]`` for a nuclear-only network) load but cannot be used with
            `predict`.

        """
        # Try to use common built-in accelerators (gpus, mps) by default
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if model_path is None:
            model_path = _released_weights_path()

        print("Initializing model...")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self._build_network(checkpoint, n_semantic_classes)

        self.model_image_shape = (256, 256)
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.required_rank = len(self.model_image_shape) + 2
        self.required_channels = 2

        self.model_mpp = 0.5

    def _build_network(self, checkpoint, n_semantic_classes):
        """The ``PanopticNet`` that matches ``checkpoint``, with its weights loaded
        strictly, on ``self.device`` and in eval mode."""
        # Weights saved before the current architecture have no 'arch_version' entry and
        # need the old (version 1) network; see torch_mesmer.model.ARCH_VERSION.
        legacy = checkpoint_arch_version(checkpoint) == LEGACY_ARCH_VERSION
        if legacy:
            print("Weights are for the older 2026-06-30 architecture; using the legacy network.")

        # use_imagenet=False: every weight is overwritten by load_state_dict
        # below, so downloading the ImageNet backbone would be wasted.
        model = PanopticNet(
            crop_size=256,
            backbone='resnet50',
            use_imagenet=False,
            pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
            backbone_levels=['C3', 'C4', 'C5'],
            n_semantic_classes=n_semantic_classes,
            legacy=legacy,
        )

        model = model.to(self.device).eval()
        model.load_state_dict(checkpoint)
        return model

    def predict(self,
                image,
                image_mpp,
                batch_size=4,
                compartment='whole-cell',
                pad_mode='constant',
                return_transforms=False,
                preprocess_kwargs=None,
                postprocess_kwargs_whole_cell=None,
                postprocess_kwargs_nuclear=None):
        """Compute a segmentation mask for `image`.

        The computed segmentation mask is either a whole-cell segmentation
        or nuclear segmentation (or both) depending on the value of `compartment`.

        The steps: rescale the images to the model's 0.5 µm/pixel, normalise
        them (`mesmer_preprocess`), cut them into 256x256 tiles, run the
        network, stitch its output back together, turn it into label masks
        (`mesmer_postprocess`) and resize the masks back to the input size.

        Input images are required to be **channels first**, with 4 dimensions
        ``[batch, channel, x, y]`` and exactly 2 channels. Add a missing batch
        dimension with ``np.expand_dims(image, 0)``; convert a channels-last
        ``[batch, x, y, channel]`` array with ``np.moveaxis(image, -1, 1)``.

        Parameters
        ----------
        image : array_like with shape ``[batch, channel, x, y]``
            The image to segment. The channel dimension must be 2 where the
            first channel represents a nuclear marker, and the second channel
            represents a whole-cell (i.e. cell membrane or cytoplasmic) marker.

        image_mpp : float
            The scale of the image in microns-per-pixel.

        batch_size : int, default=4
            Number of 256x256 tiles to run through the network at once. This
            parameter controls the memory footprint of the model inference. The
            default (4) is conservative to ensure the pipeline will run on systems
            with low resources. Increasing batch_size will significantly reduce
            computation time.

        compartment : str, {"nuclear", "whole-cell", "both"}
            What type of segmentation to perform. Must be one of
            ``"whole-cell"``, ``"nuclear"``, or ``"both"``.

        pad_mode : str, default="constant"
            Type of padding to use during image pre-tiling. See `numpy.pad` for
            details.

        return_transforms : bool, default=False
            Whether to return the raw model output (i.e. prior to postprocessing)
            along with the label image.

        preprocess_kwargs : dict, default=None
            Keyword arguments for `mesmer_preprocess`: ``threshold`` (default
            True), ``percentile`` (99.9), ``normalize`` (True) and
            ``kernel_size`` (128).

        postprocess_kwargs_whole_cell : dict, default=None
            Dictionary of keyword arguments to forward to the model post-processing
            step. See `mesmer_postprocess` for details. Keys given here override
            ``DEFAULT_POSTPROCESS_KWARGS_WHOLE_CELL``; ``None`` uses those defaults.

        postprocess_kwargs_nuclear : dict, default=None
            Dictionary of keyword arguments to forward to the model post-processing
            step. See `mesmer_postprocess` for details. Keys given here override
            ``DEFAULT_POSTPROCESS_KWARGS_NUCLEAR``; ``None`` uses those defaults.

        Returns
        -------
        label_image : array_like
            The segmentation mask(s), as integers, with:

            - Nuclear segmentation with shape ``(batch, 1, x, y)`` if ``compartment="nuclear"``
            - Whole-cell segmentation with shape ``(batch, 1, x, y)`` if ``compartment="whole-cell"``
            - Nuclear + whole-cell segmentation with shape ``(batch, 2, x, y)`` if
              ``compartment="both"``. The first channel is the nuclear segmentation and
              the second is the whole-cell segmentation, the same order as the input
              channels.

        OR

        label_image, output_image : array_like
            If ``return_transforms=True``, then the un-processed model outputs are
            also returned. These are the raw outputs from the semantic heads and have
            shape ``(batch, N, x', y')``, where ``N`` is the sum of `n_semantic_classes`,
            at the model's resolution (not resized back to the input's).
            For Mesmer, ``N = sum([1, 3, 1, 3]) = 8`` with:

            - Channels 0-3: nuclear predictions
            - Channels 4-7: whole-cell predictions
            - Within each group of 4: inner distance, then the pixelwise
              probabilities for edge, interior and background.

        Raises
        ------
        ValueError
            If `image` does not have 4 dimensions (batch, channel, width, height),
            or does not have 2 channels in dimension 1.

        Examples
        --------
        >>> X_test = np.random.random((1, 2, 512, 512))
        >>> output_image = app.predict(X_test, image_mpp=0.5, compartment='both')
        >>> print(output_image.shape)
        (1, 2, 512, 512)
        """

        image = np.asarray(image)
        self._check_input(image)

        # overwrite defaults with any user-provided values
        postprocess_kwargs_whole_cell = {**DEFAULT_POSTPROCESS_KWARGS_WHOLE_CELL,
                                         **(postprocess_kwargs_whole_cell or {})}

        postprocess_kwargs_nuclear = {**DEFAULT_POSTPROCESS_KWARGS_NUCLEAR,
                                      **(postprocess_kwargs_nuclear or {})}

        # Keep track of original shape for rescaling after processing
        orig_img_shape = image.shape

        resized_image = resize_input(image, image_mpp, self.model_mpp)
        image = mesmer_preprocess(resized_image, **(preprocess_kwargs or {}))
        tiles, tiles_info = tile_input(image, pad_mode=pad_mode, model_image_shape=self.model_image_shape)
        output_tiles = self._run_network(tiles, batch_size)
        output_images = untile_output(output_tiles, tiles_info)

        label_image = mesmer_postprocess(
                                        output_images,
                                        compartment=compartment,
                                        whole_cell_kwargs=postprocess_kwargs_whole_cell,
                                        nuclear_kwargs = postprocess_kwargs_nuclear
                                        )

        label_image = resize_output(label_image, orig_img_shape).astype(int)

        if return_transforms:
            return label_image, output_images
        else:
            return label_image

    def _check_input(self, image):
        """Raise a ``ValueError``, with a hint on how to fix it, unless ``image`` has
        shape ``(batch, 2, x, y)``."""
        if image.ndim != self.required_rank:
            raise ValueError(
                f'Input data must have {self.required_rank} dimensions '
                f'[batch, channel, x, y]; got shape {image.shape}. '
                'Add a batch dimension with np.expand_dims(image, 0).')
        if image.shape[1] != self.required_channels:
            hint = ''
            if image.shape[-1] == self.required_channels:
                hint = (' This looks like channels-last input; convert it with '
                        'np.moveaxis(image, -1, 1).')
            raise ValueError(
                f'Input data must have {self.required_channels} channels '
                f'(nuclear, membrane) in dimension 1 (channels first); '
                f'got shape {image.shape}.{hint}')

    def _run_network(self, tiles, batch_size):
        """Run the network on ``tiles``, ``(n_tiles, 2, 256, 256)``, ``batch_size`` at a time.

        Returns:
            numpy.array: the network output, ``(n_tiles, 8, 256, 256)``, in the dtype of ``tiles``.
        """
        output_tiles = np.zeros(
            (tiles.shape[0], _MESMER_OUTPUT_CHANNELS) + self.model_image_shape,
            dtype=tiles.dtype,
        )

        for tile_batch_start in range(0, tiles.shape[0], batch_size):
            # Load only this batch to GPU
            tile_batch = torch.tensor(tiles[tile_batch_start:tile_batch_start+batch_size]).to(self.device)

            with torch.inference_mode():
                pred = self.model(tile_batch)

            # Move predictions back to CPU to save GPU memory
            output_tiles[tile_batch_start:tile_batch_start+batch_size] = pred.cpu()

        return output_tiles
