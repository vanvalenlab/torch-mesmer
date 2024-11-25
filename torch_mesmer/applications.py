
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
"""Base class for applications"""


import logging
import timeit
import torch

import numpy as np

# from deepcell_toolbox.utils import resize, tile_image, untile_image
from .toolbox_utils import resize, tile_image, untile_image


class Application:
    """Application object that takes a model with weights
    and manages predictions

    Args:
        model (torch.nn.Module): ``torch.nn.Module``
            with loaded weights.
        model_image_shape (tuple): Shape of input expected by ``model``.
        dataset_metadata (str or dict): Metadata for the data that
            ``model`` was trained on.
        model_metadata (str or dict): Training metadata for ``model``.
        model_mpp (float): Microns per pixel resolution of the
            training data used for ``model``.
        preprocessing_fn (function): Pre-processing function to apply
            to data prior to prediction.
        postprocessing_fn (function): Post-processing function to apply
            to data after prediction.
            Must accept an input of a list of arrays and then
            return a single array.
        format_model_output_fn (function): Convert model output
            from a list of matrices to a dictionary with keys for
            each semantic head.

    Raises:
        ValueError: ``preprocessing_fn`` must be a callable function
        ValueError: ``postprocessing_fn`` must be a callable function
        ValueError: ``model_output_fn`` must be a callable function
    """

    def __init__(self,
                 model,
                 model_image_shape=(128, 128, 1),
                 model_mpp=0.65,
                 preprocessing_fn=None,
                 postprocessing_fn=None,
                 format_model_output_fn=None,
                 dataset_metadata=None,
                 model_metadata=None,
                 device=None):

        if device is None:
            assert(False)

        self.device = device
        self.model = model

        self.model_image_shape = model_image_shape
        # Require dimension 1 larger than model_input_shape due to addition of batch dimension
        self.required_rank = len(self.model_image_shape) + 1
        self.required_channels = self.model_image_shape[-1]

        self.model_mpp = model_mpp
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn
        self.format_model_output_fn = format_model_output_fn
        self.dataset_metadata = dataset_metadata
        self.model_metadata = model_metadata

        self.logger = logging.getLogger(self.__class__.__name__)

        # Test that pre and post processing functions are callable
        if self.preprocessing_fn is not None and not callable(self.preprocessing_fn):
            raise ValueError('Preprocessing_fn must be a callable function.')
        if self.postprocessing_fn is not None and not callable(self.postprocessing_fn):
            raise ValueError('Postprocessing_fn must be a callable function.')
        if self.format_model_output_fn is not None and not callable(self.format_model_output_fn):
            raise ValueError('Format_model_output_fn must be a callable function.')

    def predict(self, x):
        raise NotImplementedError

    def _tile_input(self, image, pad_mode='constant'):
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
            raise ValueError('deepcell_toolbox.tile_image only supports 4d images.'
                             f'Image submitted for predict has {len(image.shape)} dimensions')

        # Check difference between input and model image size
        x_diff = image.shape[1] - self.model_image_shape[0]
        y_diff = image.shape[2] - self.model_image_shape[1]

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
            tiles, tiles_info = tile_image(image, model_input_shape=self.model_image_shape,
                                           stride_ratio=0.75, pad_mode=pad_mode)

        return tiles, tiles_info

    def _untile_output(self, output_tiles, tiles_info):
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
                out = untile_image(im, tiles_info, model_input_shape=self.model_image_shape)
                return out

        if isinstance(output_tiles, list):
            output_images = [_process(o, tiles_info) for o in output_tiles]
        else:
            output_images = _process(output_tiles, tiles_info)

        return output_images

    def _batch_predict(self, tiles, batch_size):
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

            self.model.eval()

            with torch.no_grad():
                temp_input = np.transpose(batch_inputs, (0, 3, 1, 2))
                temp_input = torch.tensor(temp_input).to(self.device)
                outs = self.model(temp_input)
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

    
    def _predict_segmentation(self,
                              image,
                              batch_size=4,
                              image_mpp=None,
                              pad_mode='constant',
                              preprocess_kwargs={},
                              postprocess_kwargs={}):
        """Generates a labeled image of the input running prediction with
        appropriate pre and post processing functions.

        Input images are required to have 4 dimensions
        ``[batch, x, y, channel]``. Additional empty dimensions can be added
        using ``np.expand_dims``.

        Args:
            image (numpy.array): Input image with shape
                ``[batch, x, y, channel]``.
            batch_size (int): Number of images to predict on per batch.
            image_mpp (float): Microns per pixel for ``image``.
            pad_mode (str): The padding mode, one of "constant" or "reflect".
            preprocess_kwargs (dict): Keyword arguments to pass to the
                pre-processing function.
            postprocess_kwargs (dict): Keyword arguments to pass to the
                post-processing function.

        Raises:
            ValueError: Input data must match required rank, calculated as one
                dimension more (batch dimension) than expected by the model.

            ValueError: Input data must match required number of channels.

        Returns:
            numpy.array: Labeled image
        """
        # Check input size of image
        if len(image.shape) != self.required_rank:
            raise ValueError(f'Input data must have {self.required_rank} dimensions. '
                             f'Input data only has {len(image.shape)} dimensions')

        if image.shape[-1] != self.required_channels:
            raise ValueError(f'Input data must have {self.required_channels} channels. '
                             f'Input data only has {image.shape[-1]} channels')

        # Tile images, raises error if the image is not 4d
        tiles, tiles_info = self._tile_input(image, pad_mode=pad_mode)

        # Run images through model
        t = timeit.default_timer()
        output_tiles = self._batch_predict(tiles=tiles, batch_size=batch_size)
        self.logger.debug('Model inference finished in %s s',
                          timeit.default_timer() - t)

        # Untile images
        output_images = self._untile_output(output_tiles, tiles_info)

        return output_images
