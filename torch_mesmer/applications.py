
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