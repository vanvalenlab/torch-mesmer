# Copyright 2016-2024 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mesmer application"""

import logging

import numpy as np

import torch

from .deep_watershed import deep_watershed
from .toolbox_processing import percentile_threshold
from .toolbox_processing import histogram_normalization

from .toolbox_utils import resize, tile_image, untile_image


# TODO: Replace with torch model
MODEL_KEY = 'models/MultiplexSegmentation-9.tar.gz'
MODEL_NAME = 'MultiplexSegmentation'
MODEL_HASH = 'a1dfbce2594f927b9112f23a0a1739e0'


# pre- and post-processing functions
def mesmer_preprocess(image, **kwargs):
    """Preprocess input data for Mesmer model.

    Args:
        image: array to be processed

    Returns:
        np.array: processed image array
    """

    if len(image.shape) != 4:
        raise ValueError(f"Image data must be 4D, got image of shape {image.shape}")

    output = np.copy(image)
    threshold = kwargs.get('threshold', True)
    if threshold:
        percentile = kwargs.get('percentile', 99.9)
        output = percentile_threshold(image=output, percentile=percentile)

    normalize = kwargs.get('normalize', True)
    if normalize:
        kernel_size = kwargs.get('kernel_size', 128)
        output = histogram_normalization(image=output, kernel_size=kernel_size)

    return output


def format_output_mesmer(output_list):
    """Takes list of model outputs and formats into a dictionary for better readability

    Args:
        output_list (list): predictions from semantic heads

    Returns:
        dict: Dict of predictions for whole cell and nuclear.

    Raises:
        ValueError: if model output list is not len(4)
    """
    expected_length = 4
    if len(output_list) != expected_length:
        raise ValueError('output_list was length {}, expecting length {}'.format(
            len(output_list), expected_length))

    formatted_dict = {
        'whole-cell': [output_list[0], output_list[1][..., 1:2]],
        'nuclear': [output_list[2], output_list[3][..., 1:2]],
    }

    return formatted_dict


def mesmer_postprocess(model_output, compartment='whole-cell',
                       whole_cell_kwargs=None, nuclear_kwargs=None):
    """Postprocess Mesmer output to generate predictions for distinct cellular compartments

    Args:
        model_output (dict): Output from the Mesmer model. A dict with a key corresponding to
            each cellular compartment with a model prediction. Each key maps to a subsequent dict
            with the following keys entries
            - inner-distance: Prediction for the inner distance transform.
            - outer-distance: Prediction for the outer distance transform
            - fgbg-fg: prediction for the foreground/background transform
            - pixelwise-interior: Prediction for the interior/border/background transform.
        compartment: which cellular compartments to generate predictions for.
            must be one of 'whole_cell', 'nuclear', 'both'
        whole_cell_kwargs (dict): Optional list of post-processing kwargs for whole-cell prediction
        nuclear_kwargs (dict): Optional list of post-processing kwargs for nuclear prediction

    Returns:
        numpy.array: Uniquely labeled mask for each compartment

    Raises:
        ValueError: for invalid compartment flag
    """

    valid_compartments = ['whole-cell', 'nuclear', 'both']
    
    if whole_cell_kwargs is None:
        whole_cell_kwargs = {}

    if nuclear_kwargs is None:
        nuclear_kwargs = {}

    if compartment not in valid_compartments:
        raise ValueError(f'Invalid compartment supplied: {compartment}. '
                         f'Must be one of {valid_compartments}')

    if compartment == 'whole-cell':
        label_images = deep_watershed(model_output['whole-cell'],
                                      **whole_cell_kwargs)
    elif compartment == 'nuclear':
        label_images = deep_watershed(model_output['nuclear'],
                                      **nuclear_kwargs)
    elif compartment == 'both':
        label_images_cell = deep_watershed(model_output['whole-cell'],
                                           **whole_cell_kwargs)

        label_images_nucleus = deep_watershed(model_output['nuclear'],
                                              **nuclear_kwargs)

        label_images = np.concatenate([
            label_images_cell,
            label_images_nucleus
        ], axis=-1)

    else:
        raise ValueError(f'Invalid compartment supplied: {compartment}. '
                         f'Must be one of {valid_compartments}')

    return label_images

def resize_input(image, image_mpp, model_mpp):
    """Checks if there is a difference between image and model resolution
    and resizes if they are different. Otherwise returns the unmodified
    image.

    Args:
        image (numpy.array): Input image to resize.
        image_mpp (float): Microns per pixel for the ``image``.

    Returns:
        numpy.array: Input image resized if necessary to match ``model_mpp``
    """
    # Don't scale the image if mpp is the same or not defined
    if image_mpp not in {None, model_mpp}:
        shape = image.shape
        scale_factor = image_mpp / model_mpp
        new_shape = (int(shape[1] * scale_factor),
                        int(shape[2] * scale_factor))
        image = resize(image, new_shape, data_format='channels_last')
    return image

def resize_output(image, original_shape):
        """Rescales input if the shape does not match the original shape
        excluding the batch and channel dimensions.

        Args:
            image (numpy.array): Image to be rescaled to original shape
            original_shape (tuple): Shape of the original input image

        Returns:
            numpy.array: Rescaled image
        """
        if not isinstance(image, list):
            image = [image]

        for i in range(len(image)):
            img = image[i]
            # Compare x,y based on rank of image
            # Check if unnecessary
            if len(img.shape) == 4:
                same = img.shape[1:-1] == original_shape[1:-1]
            elif len(img.shape) == 3:
                same = img.shape[1:] == original_shape[1:-1]
            else:
                same = img.shape == original_shape[1:-1]

            # Resize if same is false
            if not same:
                # Resize function only takes the x,y dimensions for shape
                new_shape = original_shape[1:-1]
                img = resize(img, new_shape,
                             data_format='channels_last',
                             labeled_image=True)
            image[i] = img

        if len(image) == 1:
            image = image[0]

        return image

def tile_input(image, model_image_shape, pad_mode='constant'):
    """Tile the input image to match shape expected by model
    using the ``deepcell_toolbox`` or ``toolbox_utils`` function.

    Only supports 4D images.

    Args:
        image (numpy.array): Input image to tile
        pad_mode (str): The padding mode, one of "constant" or "reflect".

    Raises:
        ValueError: Input images must have only 4 dimensions

    Returns:
        (numpy.array, dict): Tuple of tiled image and dict of tiling
        information.
    """
    if len(image.shape) != 4:
        raise ValueError('toolbox_utils.tile_image only supports 4d images.'
                            f'Image submitted for predict has {len(image.shape)} dimensions')

    # Check difference between input and model image size
    x_diff = image.shape[1] - model_image_shape[0]
    y_diff = image.shape[2] - model_image_shape[1]

    # Check if the input is smaller than model image size
    if x_diff < 0 or y_diff < 0:
        # Calculate padding
        x_diff, y_diff = abs(x_diff), abs(y_diff)
        x_pad = (x_diff // 2, x_diff // 2 + 1) if x_diff % 2 else (x_diff // 2, x_diff // 2)
        y_pad = (y_diff // 2, y_diff // 2 + 1) if y_diff % 2 else (y_diff // 2, y_diff // 2)

        tiles = np.pad(image, [(0, 0), x_pad, y_pad, (0, 0)], 'reflect')
        tiles_info = {'padding': True,
                        'x_pad': x_pad,
                        'y_pad': y_pad}
    # Otherwise tile images larger than model size
    else:
        # Tile images, needs 4d
        tiles, tiles_info = tile_image(image, model_input_shape=model_image_shape,
                                        stride_ratio=0.75, pad_mode=pad_mode)

    return tiles, tiles_info

def batch_predict(tiles, batch_size, model, device):
    """Batch process tiles to generate model predictions.

    Batch processing occurs without loading entire image stack onto 
    GPU memory, a problem that exists in other solutions such as
    keras.predict.

    Args:
        tiles (numpy.array): Tiled data which will be fed to model
        batch_size (int): Number of images to predict on per batch

    Returns:
        list: Model outputs
    """

    # list to hold final output
    output_tiles = []

    # loop through each batch
    for i in range(0, tiles.shape[0], batch_size):
        batch_inputs = tiles[i:i + batch_size, ...]

        model.eval()

        with torch.no_grad():
            temp_input = np.transpose(batch_inputs, (0, 3, 1, 2))
            temp_input = torch.tensor(temp_input).to(device)
            outs = model(temp_input)
            batch_outputs = [torch.permute(i, (0, 2, 3, 1)) for i in outs]

        # model with only a single output gets temporarily converted to a list
        if not isinstance(batch_outputs, list):
            batch_outputs = [batch_outputs.cpu().detach()]

        else:
            batch_outputs = [b_out.cpu().detach() for b_out in batch_outputs]

        # initialize output list with empty arrays to hold all batches
        if not output_tiles:
            for batch_out in batch_outputs:
                shape = (tiles.shape[0],) + batch_out.shape[1:]
                output_tiles.append(np.zeros(shape, dtype=tiles.dtype))

        # save each batch to corresponding index in output list
        for j, batch_out in enumerate(batch_outputs):
            output_tiles[j][i:i + batch_size, ...] = batch_out

    return output_tiles

def untile_output(output_tiles, tiles_info, model_image_shape):
    """Untiles either a single array or a list of arrays
    according to a dictionary of tiling specs

    Args:
        output_tiles (numpy.array or list): Array or list of arrays.
        tiles_info (dict): Tiling specs output by the tiling function.

    Returns:
        numpy.array or list: Array or list according to input with untiled images
    """
    # If padding was used, remove padding
    if tiles_info.get('padding', False):
        def _process(im, tiles_info):
            ((xl, xh), (yl, yh)) = tiles_info['x_pad'], tiles_info['y_pad']
            # Edge-case: upper-bound == 0 - this can occur when only one of
            # either X or Y is smaller than model_img_shape while the other
            # is equal to model_image_shape.
            xh = -xh if xh != 0 else None
            yh = -yh if yh != 0 else None
            return im[:, xl:xh, yl:yh, :]
    # Otherwise untile
    else:
        def _process(im, tiles_info):
            out = untile_image(im, tiles_info, model_input_shape=model_image_shape)
            return out

    if isinstance(output_tiles, list):
        output_images = [_process(o, tiles_info) for o in output_tiles]
    else:
        output_images = _process(output_tiles, tiles_info)

    return output_images

class Mesmer():
    """Loads a :mod:`torch-mesmer.panopticnet.PanopticNet` model for
    tissue segmentation with pretrained weights.

    The ``predict`` method handles prep and post processing steps
    to return a labeled image.

    Example:

    .. code-block:: python

        from skimage.io import imread
        from torch_mesmer.mesmer import Mesmer

        # Load the images
        im1 = imread('TNBC_DNA.tiff')
        im2 = imread('TNBC_Membrane.tiff')

        # Combined together and expand to 4D
        im = np.stack((im1, im2), axis=-1)
        im = np.expand_dims(im,0)

        # Create the application
        app = Mesmer()

        # create the lab
        labeled_image = app.predict(image)

    Args:
        model (torch.nn.Module): The model to load. If ``None``,
            a pre-trained model will be downloaded.
        device (torch.device): The device to run model on.
    """

    #: Metadata for the dataset used to train the model
    dataset_metadata = {
        'name': '20200315_IF_Training_6.npz',
        'other': 'Pooled whole-cell data across tissue types'
    }

    #: Metadata for the model and training process
    model_metadata = {
        'batch_size': 1,
        'lr': 1e-5,
        'lr_decay': 0.99,
        'training_seed': 0,
        'n_epochs': 30,
        'training_steps_per_epoch': 1739 // 1,
        'validation_steps_per_epoch': 193 // 1
    }

    def __init__(self, model=None, device=torch.device("cpu")):

        if model is None:
            raise Exception("Need to provide a model")

            # # Uncomment with saved model later

            # cache_subdir = "models"
            # model_dir = Path.home() / ".deepcell" / "models"
            # # archive_path = fetch_data(
            # #     asset_key=MODEL_KEY,
            # #     cache_subdir=cache_subdir,
            # #     file_hash=MODEL_HASH
            # # )
            # # extract_archive(archive_path, model_dir)
            # model_path = model_dir / MODEL_NAME
            # model = tf.keras.models.load_model(model_path)

        self.device = device
        self.model = model

        self.model_image_shape = model.input_shape[1:]
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.required_rank = len(self.model_image_shape) + 1
        self.required_channels = self.model_image_shape[-1]

        self.model_mpp = 0.5

        # # We can choose to bind these functions to the object or 
        # # we can just have them global, should be the same

        # self.preprocessing_fn = preprocessing_fn
        # self.postprocessing_fn = postprocessing_fn
        # self.format_model_output_fn = format_model_output_fn
        # # Test that pre and post processing functions are callable
        # if self.preprocessing_fn is not None and not callable(self.preprocessing_fn):
        #     raise ValueError('Preprocessing_fn must be a callable function.')
        # if self.postprocessing_fn is not None and not callable(self.postprocessing_fn):
        #     raise ValueError('Postprocessing_fn must be a callable function.')
        # if self.format_model_output_fn is not None and not callable(self.format_model_output_fn):
        #     raise ValueError('Format_model_output_fn must be a callable function.')

        
        # Not used properly right now
        self.logger = logging.getLogger(self.__class__.__name__)

    def predict(self,
                image,
                batch_size=4,
                image_mpp=None,
                preprocess_kwargs={},
                compartment='whole-cell',
                pad_mode='constant',
                postprocess_kwargs_whole_cell={},
                postprocess_kwargs_nuclear={}):
        """Generates a labeled image of the input running prediction with
        appropriate pre and post processing functions.

        Input images are required to have 4 dimensions
        ``[batch, x, y, channel]``.
        Additional empty dimensions can be added using ``np.expand_dims``.

        Args:
            image (numpy.array): Input image with shape
                ``[batch, x, y, channel]``.
            batch_size (int): Number of images to predict on per batch.
            image_mpp (float): Microns per pixel for ``image``.
            compartment (str): Specify type of segmentation to predict.
                Must be one of ``"whole-cell"``, ``"nuclear"``, ``"both"``.
            preprocess_kwargs (dict): Keyword arguments to pass to the
                pre-processing function.
            postprocess_kwargs (dict): Keyword arguments to pass to the
                post-processing function.

        Raises:
            ValueError: Input data must match required rank of the application,
                calculated as one dimension more (batch dimension) than expected
                by the model.

            ValueError: Input data must match required number of channels.

        Returns:
            numpy.array: Instance segmentation mask.
        """
        default_kwargs_cell = {
            'maxima_threshold': 0.075,
            'maxima_smooth': 0,
            'interior_threshold': 0.2,
            'interior_smooth': 2,
            'small_objects_threshold': 15,
            'fill_holes_threshold': 15,
            'radius': 2
        }

        default_kwargs_nuc = {
            'maxima_threshold': 0.1,
            'maxima_smooth': 0,
            'interior_threshold': 0.2,
            'interior_smooth': 2,
            'small_objects_threshold': 15,
            'fill_holes_threshold': 15,
            'radius': 2
        }

        # overwrite defaults with any user-provided values
        postprocess_kwargs_whole_cell = {**default_kwargs_cell,
                                         **postprocess_kwargs_whole_cell}

        postprocess_kwargs_nuclear = {**default_kwargs_nuc,
                                      **postprocess_kwargs_nuclear}

        # create dict to hold all of the post-processing kwargs
        postprocess_kwargs = {
            'whole_cell_kwargs': postprocess_kwargs_whole_cell,
            'nuclear_kwargs': postprocess_kwargs_nuclear,
            'compartment': compartment
        }

        resized_image = resize_input(image, image_mpp, self.model_mpp)

        image = mesmer_preprocess(resized_image, **preprocess_kwargs)

        # Tile images, raises error if the image is not 4d
        tiles, tiles_info = tile_input(image, pad_mode=pad_mode, model_image_shape=self.model_image_shape)

        output_tiles = batch_predict(tiles=tiles, batch_size=batch_size, model=self.model, device=self.device)
        
        # Untile images
        output_images = untile_output(output_tiles, tiles_info, self.model_image_shape)
    
        output_images = format_output_mesmer(output_images)
        label_image = mesmer_postprocess(output_images, **postprocess_kwargs)
        # Restore channel dimension if not already there
        # TODO: check if unnecessary
        if len(image.shape) == self.required_rank - 1:
            image = np.expand_dims(image, axis=-1)

        label_image = resize_output(label_image, image.shape)
        return label_image
