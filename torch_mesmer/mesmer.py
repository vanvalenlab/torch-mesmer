import glob
from pathlib import Path
import torch
torch.set_num_threads(24)

import numpy as np

from torch_mesmer.model import PanopticNet

from torch_mesmer.postprocess_utils import resize_input, resize_output, mesmer_postprocess, mesmer_preprocess, untile_output, tile_input

from huggingface_hub import hf_hub_download


class Mesmer():
    """The Mesmer cell segmentation pipeline for multiplexed images.

    The input to this pipeline is a multiplexed image comprising either a single
    channel or two channels:

    - If a single channel, then the channel should represent a nuclear marker and
      nuclear segmentation is performed (with ``compartment="nuclear"``.
    - If two channels, then the first channel should represent a nuclear marker and
      the second channel a cell membrane or cytosol marker. It is then possible to
      perform whole-cell segmentation with ``compartment="whole-cell"`` or both
      nuclear and cell segmentation simultaneously with ``compartment="both"``

    See the `predict` docstring for details.
    """
    def __init__(
        self, *, model_path=None, device=None, n_semantic_classes=[1, 3, 1, 3]
    ):
        """        
        Instantiate an instance of the Mesmer cell segmentation pipeline.

        The returned instance is designed for segmenting whole-slide images.

        Parameters
        ----------
        model_path : str or pathlib.Path, default=None
            The path to the trained model (i.e. a ``.pth`` file).
            If not specified (the default), an attempt will be made to download
            the latest model weights from huggingface.

            .. note::
               Internet access and a valid ``HF_TOKEN`` is required to download
               the latest weights. See :doc:`/model_access` for details.

        device : str, default=None
            A `torch.device` compatible specifier indicating the hardware to be
            used for inference, e.g. ``"cuda"``, ``"mps"``, or ``"cpu"``.
            If not specified (the default), Mesmer will use a cuda or mps-compatible
            backend if it is detected, falling back to ``"cpu"`` if not.
        
        n_semantic_classes : list of int, default=[1, 3, 1, 3]
            Number of prediction heads used in the model. For Mesmer, use ``[1, 3, 1, 3]``,
            for Dynamic Nuclear Net, use ``[1, 3]``.

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
            
            hf_hub_download(repo_id='vanvalenlab/torch-mesmer', 
                            filename='torch-mesmer_2026-06-30.pth',
                            local_dir=Path.home() / '.deepcell/models')
            model_path = Path.home() / '.deepcell/models/torch-mesmer_2026-06-30.pth'

        print("Initializing model...")
        
        self.model = PanopticNet(
            crop_size=256,
            backbone='resnet50',
            pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
            backbone_levels=['C3', 'C4', 'C5'],
            n_semantic_classes=n_semantic_classes
        )

        self.model = self.model.to(self.device).eval()

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        # Whole cell, nuc

        self.model_image_shape = (256, 256)
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.required_rank = len(self.model_image_shape) + 2
        self.required_channels = 2

        self.model_mpp = 0.5
        
    def predict(self,
                image,
                image_mpp,
                *,
                batch_size=4,
                compartment='whole-cell',
                pad_mode='constant',
                return_transforms=False,
                postprocess_kwargs_whole_cell={},
                postprocess_kwargs_nuclear={}):
        
        """Compute a segmentation mask for `image`.

        The computed segmentation mask is either a whole-cell segmentation
        or nuclear segmentation (or both) depending on the value of `compartment`.

        Input images are required to have 4 dimensions ``[batch, channel, x, y]``.
        Channel dimension must be 2 and must come first.

        Parameters
        ----------
        image : array_like with shape ``[batch, channel, x, y]``
            The image to segment. The channel dimension must be 2 where the
            first channel represents a nuclear marker, and the second channel
            represents a whole-cell (i.e. cell membrane or cytoplasmic) marker

        image_mpp : float
            The scale of the image in microns-per-pixel.

        batch_size : int, default=4
            Number of images to predict per batch. This parameter controls the
            memory footprint of the model inference. The default (4) is
            conservative to ensure the pipeline will run on systems with low
            resources. Increasing batch_size will significantly reduce computation
            time.

        compartment : str, {"nuclear", "whole-cell", "both"}
            What type of segmentation to perform. Must be one of
            ``"whole-cell"``, ``"nuclear"``, or ``"both"``.

        pad_mode : str, default="constant"
            Type of padding to use during image pre-tiling. See `numpy.pad` for
            details.

        return_transforms : bool, default=False
            Whether to return the raw model output (i.e. prior to postprocessing)
            along with the label image.

        postprocess_kwargs_whole_cell : dict, default={}
            Dictionary of keyword arguments to forward to the model post-processing
            step. See `mesmer_postprocess` for details.

        postprocess_kwargs_nuclear : dict, default={}
            Dictionary of keyword arguments to forward to the model post-processing
            step. See `mesmer_postprocess` for details.

        Returns
        -------
        label_image : array_like
            The segmentation mask(s) with:

            - Nuclear segmentation with shape ``(batch, 1, x, y)`` if ``compartment="nuclear"``
            - Whole-cell segmentation with shape ``(batch, 1, x, y)`` if ``compartment="whole-cell"``
            - Nuclear + whole-cell segmentation with shape ``(batch, 2, x, y)`` if
              ``compartment="both"``. The first channel is the nuclear segmentation and
              the second is the whole-cell segmentation.

              .. caution:
                 Note that this is the opposite of the input order!


        OR

        label_image, output_image : array_like
            If ``return_transforms=True``, then the un-processed model outputs are
            also returned. These are the raw outputs from the semantic heads and have
            shape ``(batch, x, y, N)``, where ``N`` is the sum of `n_semantic_classes`.
            For Mesmer, ``N = sum([1, 3, 1, 3]) = 8`` with:

            - Channels 1-4 (inds 0-3): cytoplasmic predictions
            - Channels 5-8 (inds 4-7): nuclear predictions
            - Prediction 1: Inner distance transform
            - Prediction 2: Outer boundary of the object
            - Prediction 3: Interior pixels of the object
            - Prediction 4: Image background

        Raises
        ------
        ValueError
            If `image` does not have 4 dimensions (batch, channel, width, height)
            If the image does not have the correct number of channels (1 for nuclear
            segmentation, 2 for whole-cell).
        """



        default_kwargs_nuc = {
            'maxima_threshold': 0.1,
            'maxima_smooth': 0,
            'interior_threshold': 0.075,
            'interior_smooth': 1,
            'small_objects_threshold': 15,
            'fill_holes_threshold': 15,
            'radius': 2,
            'maxima_index': 0,
            'interior_index': 2,
            'pixel_expansion': 0
        }

        default_kwargs_cell = {
            'maxima_threshold': 0.1,
            'maxima_smooth': 1,
            'interior_threshold': 0.075,
            'interior_smooth': 1,
            'small_objects_threshold': 15,
            'fill_holes_threshold': 15,
            'radius': 2,
            'maxima_index': 4,
            'interior_index': 6,
            'pixel_expansion': 0
        }

        # overwrite defaults with any user-provided values
        postprocess_kwargs_whole_cell = {**default_kwargs_cell,
                                         **postprocess_kwargs_whole_cell}

        postprocess_kwargs_nuclear = {**default_kwargs_nuc,
                                      **postprocess_kwargs_nuclear}

        # Keep track of original shape for rescaling after processing
        orig_img_shape = image.shape

        resized_image = resize_input(image, image_mpp, self.model_mpp)
        image = mesmer_preprocess(resized_image)
        # Tile images, raises error if the image is not 4d
        tiles, tiles_info = tile_input(image, pad_mode=pad_mode, model_image_shape=self.model_image_shape)
        B_tiles = tiles.shape[0]
        output_tiles = np.zeros(
            (B_tiles,) + (8,) + self.model_image_shape,
            dtype=tiles.dtype,
        )

        for tile_batch_start in range(0, tiles.shape[0], batch_size):
            # Load only this batch to GPU
            tile_batch = torch.tensor(tiles[tile_batch_start:tile_batch_start+batch_size]).to(self.device)
            
            with torch.inference_mode():
                pred = self.model(tile_batch)
                        
            # Move predictions back to CPU to save GPU memory
            output_tiles[tile_batch_start:tile_batch_start+batch_size] = pred.cpu()

        # Untile images
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
