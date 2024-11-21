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
"""Semantic segmentation data generators."""


import os

import numpy as np
from skimage.transform import rescale, resize

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.platform import tf_logging as logging

try:
    import scipy
    # scipy.linalg cannot be accessed until explicitly imported
    from scipy import linalg
except ImportError:
    scipy = None

from deepcell.image_generators import _transform_masks


class SemanticIterator(Iterator):
    """Iterator yielding data from Numpy arrays (``X`` and ``y``).

    Args:
        train_dict (dict): Consists of numpy arrays for ``X`` and ``y``.
        image_data_generator (ImageDataGenerator): For random transformations
            and normalization.
        batch_size (int): Size of a batch.
        min_objects (int): Images with fewer than ``min_objects`` are ignored.
        shuffle (bool): Whether to shuffle the data between epochs.
        seed (int): Random seed for data shuffling.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        save_to_dir (str): Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix (str): Prefix to use for saving sample
            images (if ``save_to_dir`` is set).
        save_format (str): Format to use for saving sample images
            (if ``save_to_dir`` is set).
    """
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 batch_size=1,
                 shuffle=False,
                 transforms=['outer-distance'],
                 transforms_kwargs={},
                 seed=None,
                 min_objects=3,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        # Load data
        if 'X' not in train_dict:
            raise ValueError('No training data found in train_dict')

        if 'y' not in train_dict:
            raise ValueError('Instance masks are required for the '
                             'SemanticIterator')

        X, y = train_dict['X'], train_dict['y']

        if X.shape[0] != y.shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             f'length. Found X.shape: {X.shape} y.shape: {y.shape}')

        if X.ndim != 4:
            raise ValueError('Input data in `SemanticIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', X.shape)

        self.x = np.asarray(X, dtype=K.floatx())
        self.y = np.asarray(y, dtype='int32')

        self.transforms = transforms
        self.transforms_kwargs = transforms_kwargs

        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.min_objects = min_objects

        # Remove images with small numbers of cells
        invalid_batches = []
        for b in range(self.x.shape[0]):
            if len(np.unique(self.y[b])) - 1 < self.min_objects:
                invalid_batches.append(b)

        invalid_batches = np.array(invalid_batches, dtype='int')

        if invalid_batches.size > 0:
            logging.warning('Removing %s of %s images with fewer than %s '
                            'objects.', invalid_batches.size, self.x.shape[0],
                            self.min_objects)

        self.x = np.delete(self.x, invalid_batches, axis=0)
        self.y = np.delete(self.y, invalid_batches, axis=0)

        super().__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _transform_labels(self, y):
        y_semantic_list = []
        # loop over channels axis of labels in case there are multiple label types
        for label_num in range(y.shape[self.channel_axis]):

            if self.channel_axis == 1:
                y_current = y[:, label_num:label_num + 1, ...]
            else:
                y_current = y[..., label_num:label_num + 1]

            for transform in self.transforms:
                transform_kwargs = self.transforms_kwargs.get(transform, dict())
                y_transform = _transform_masks(y_current, transform,
                                               data_format=self.data_format,
                                               **transform_kwargs)
                y_semantic_list.append(y_transform)

        return y_semantic_list

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]),
                           dtype=self.x.dtype)
        batch_y = []

        for i, j in enumerate(index_array):
            x = self.x[j]

            # _transform_labels expects batch dimension
            y_semantic_list = self._transform_labels(self.y[j:j + 1])

            # initialize batch_y
            if len(batch_y) == 0:
                for ys in y_semantic_list:
                    shape = tuple([len(index_array)] + list(ys.shape[1:]))
                    batch_y.append(np.zeros(shape, dtype=ys.dtype))

            # random_transform does not expect batch dimension
            y_semantic_list = [ys[0] for ys in y_semantic_list]

            # Apply transformation
            x, y_semantic_list = self.image_data_generator.random_transform(
                x, y_semantic_list)

            x = self.image_data_generator.standardize(x)

            batch_x[i] = x

            for k, ys in enumerate(y_semantic_list):
                batch_y[k][i] = ys

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                if self.data_format == 'channels_first':
                    img_x = np.expand_dims(batch_x[i, 0, ...], 0)
                else:
                    img_x = np.expand_dims(batch_x[i, ..., 0], -1)
                img = array_to_img(img_x, self.data_format, scale=True)
                fname = f'{self.save_prefix}_{j}_{np.random.randint(1e4)}.{self.save_format}'
                img.save(os.path.join(self.save_to_dir, fname))

                if self.y is not None:
                    # Save argmax of y batch
                    for k, y_sem in enumerate(batch_y):
                        if y_sem[i].shape[self.channel_axis - 1] == 1:
                            img_y = y_sem[i]
                        else:
                            img_y = np.argmax(y_sem[i],
                                              axis=self.channel_axis - 1)
                            img_y = np.expand_dims(img_y,
                                                   axis=self.channel_axis - 1)
                        img = array_to_img(img_y, self.data_format, scale=True)
                        fname = 'y_{sem}_{prefix}_{index}_{hash}.{format}'.format(
                            sem=k,
                            prefix=self.save_prefix,
                            index=j,
                            hash=np.random.randint(1e4),
                            format=self.save_format)
                        img.save(os.path.join(self.save_to_dir, fname))

        return batch_x, batch_y

    def next(self):
        """For python 2.x. Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class SemanticDataGenerator(ImageDataGenerator):
    """Generates batches of tensor image data with real-time data augmentation.
    The data will be looped over (in batches).

    Args:
        featurewise_center (bool): Set input mean to 0 over the dataset,
            feature-wise.
        samplewise_center (bool): Set each sample mean to 0.
        featurewise_std_normalization (bool): Divide inputs by std
            of the dataset, feature-wise.
        samplewise_std_normalization (bool): Divide each input by its std.
        zca_epsilon (float): Epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening (bool): Apply ZCA whitening.
        rotation_range (int): Degree range for random rotations.
        width_shift_range (float): 1-D array-like or int

            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
              ``(-width_shift_range, +width_shift_range)``
            - With ``width_shift_range=2`` possible values are integers
              ``[-1, 0, +1]``, same as with ``width_shift_range=[-1, 0, +1]``,
              while with ``width_shift_range=1.0`` possible values are floats
              in the interval [-1.0, +1.0).

        height_shift_range: Float, 1-D array-like or int

            - float: fraction of total height, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
              ``(-height_shift_range, +height_shift_range)``
            - With ``height_shift_range=2`` possible values
              are integers ``[-1, 0, +1]``,
              same as with ``height_shift_range=[-1, 0, +1]``,
              while with ``height_shift_range=1.0`` possible values are floats
              in the interval [-1.0, +1.0).

        shear_range (float): Shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range (float): float or [lower, upper], Range for random zoom.
            If a float, ``[lower, upper] = [1-zoom_range, 1+zoom_range]``.
        channel_shift_range (float): range for random channel shifts.
        fill_mode (str): One of {"constant", "nearest", "reflect" or "wrap"}.

            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:

                - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - 'nearest':  aaaaaaaa|abcd|dddddddd
                - 'reflect':  abcddcba|abcd|dcbaabcd
                - 'wrap':  abcdabcd|abcd|abcdabcd

        cval (float): Value used for points outside the boundaries
            when ``fill_mode = "constant"``.
        horizontal_flip (bool): Randomly flip inputs horizontally.
        vertical_flip (bool): Randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None. If None or 0, no rescaling
            is applied, otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        validation_split (float): Fraction of images reserved for validation
            (strictly between 0 and 1).
    """
    def flow(self,
             train_dict,
             batch_size=1,
             transforms=['outer-distance'],
             transforms_kwargs={},
             min_objects=3,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Args:
            train_dict (dict): Consists of numpy arrays for ``X`` and ``y``.
            batch_size (int): Size of a batch. Defaults to 1.
            shuffle (bool): Whether to shuffle the data between epochs.
                Defaults to ``True``.
            seed (int): Random seed for data shuffling.
            min_objects (int): Minumum number of objects allowed per image
            save_to_dir (str): Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix (str): Prefix to use for saving sample
                images (if ``save_to_dir`` is set).
            save_format (str): Format to use for saving sample images
                (if ``save_to_dir`` is set).

        Returns:
            SemanticIterator: An ``Iterator`` yielding tuples of ``(x, y)``,
            where ``x`` is a numpy array of image data and ``y`` is list of
            numpy arrays of transformed masks of the same shape.
        """
        return SemanticIterator(
            train_dict,
            self,
            batch_size=batch_size,
            transforms=transforms,
            transforms_kwargs=transforms_kwargs,
            shuffle=shuffle,
            min_objects=min_objects,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def random_transform(self, x, y=None, seed=None):
        """Applies a random transformation to an image.

        Args:
            x (numpy.array): 3D tensor or list of 3D tensors,
                single image.
            y (numpy.array): 3D tensor or list of 3D tensors,
                label mask(s) for ``x``, optional.
            seed (int): Random seed.

        Returns:
            numpy.array: A randomly transformed copy of the input (same shape).
            If ``y`` is passed, it is transformed if necessary and returned.
        """
        params = self.get_random_transform(x.shape, seed)

        if isinstance(x, list):
            x = [self.apply_transform(x_i, params) for x_i in x]
        else:
            x = self.apply_transform(x, params)

        if y is None:
            return x

        # Nullify the transforms that don't affect `y`
        params['brightness'] = None
        params['channel_shift_intensity'] = None
        _interpolation_order = self.interpolation_order
        self.interpolation_order = 0

        if isinstance(y, list):
            y_new = []
            for y_i in y:
                if y_i.shape[self.channel_axis - 1] > 1:
                    y_t = self.apply_transform(y_i, params)

                # Keep original interpolation order if it is a
                # regression task
                elif y_i.shape[self.channel_axis - 1] == 1:
                    self.interpolation_order = _interpolation_order
                    y_t = self.apply_transform(y_i, params)
                    self.interpolation_order = 0
                y_new.append(y_t)
            y = y_new
        else:
            y = self.apply_transform(y, params)

        self.interpolation_order = _interpolation_order
        return x, y
