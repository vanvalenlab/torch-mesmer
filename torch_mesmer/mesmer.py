import torch
torch.set_num_threads(24)

import numpy as np

from torch_mesmer.model import PanopticNet

from torch_mesmer.postprocess_utils import resize_input, resize_output, mesmer_postprocess, mesmer_preprocess, untile_output, tile_input


class Mesmer():

    def __init__(
            self, 
            model_path=None, 
            device=None, 
            n_semantic_classes=[1,3,1,3]
    ):
        
        """        
        Initializes a Panoptic network segmentation model using the following parameters.
        
        :params model_path: the path to where the model weights are stored
        :type model_path: str
        
        :params device: GPU where you would like to conduct inference. 
            Must be one of "cuda", "mps" or "cpu"
        :type device: str

        :params n_semantic_classes: Number of prediction heads used in the model.
                For Mesmer, use `[1, 3, 1, 3]` 
                for Dynamic Nuclear Net, use `[1, 3]`
        :type n_semantic_classes: list

        """

        if device is None:
            self.device = 'cpu'
        else:
            self.device=device

        if model_path is None:
            raise Exception("Please provide a path to the model checkpoint file.")
        
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
                batch_size=4,
                image_mpp=None,
                compartment='whole-cell',
                pad_mode='constant',
                return_transforms=False,
                postprocess_kwargs_whole_cell={},
                postprocess_kwargs_nuclear={}):
        
        """Generates a labeled image of the input running prediction with
        appropriate pre and post processing functions.

        Input images are required to have 4 dimensions
        ``[batch, channel, x, y]``. Channel dimension must be 2 and must come first.
        Additional empty dimensions can be added using ``np.expand_dims``.

        :param image: Input image with shape
                ``[batch, channel, x, y]``. 
                **Nuclear image is expected in the first channel,
                cytoplasmic image is expected in the second.**
        :type image: (numpy.array)
        
        :param batch_size: Number of images to predict on per batch.
        :type batch_size: int

        :param image_mpp: Microns per pixel for ``image``.
        :type image_mpp: float

        :param compartment: Specify type of segmentation to predict.
                Must be one of ``"whole-cell"``, ``"nuclear"``, ``"both"``.
        :type compartment: str
        
        :param preprocess_kwargs: Keyword arguments to pass to the
                pre-processing function.
        :type preprocess_kwargs: dict

        :param postprocess_kwargs: Keyword arguments to pass to the
                pre-processing function.
        :type postprocess_kwargs: dict

        :raises ValueError: Input data must match required rank of the application,
                calculated as one dimension more (batch dimension) than expected
                by the model.
        :raises ValueError: Input data must match required number of channels.

        :returns label_image:  Instance segmentation mask with shape ``[batch, x, y, channel]``.
                Cytoplasmic mask is returned in the first channel, nuclear mask is 
                returned in the second. **Note that this is opposite of the input order!**
        :type label_image: numpy.array
        :returns output_transforms: Raw predictions from the model itself with 
                shape ``[batch, x, y, 8]``.

                - Channels 1-4 (inds 0-3): cytoplasmic predictions
                - Channels 5-8 (inds 4-7): nuclear predictions
                - Prediction 1: Inner distance transform
                - Prediction 2: Outer boundary of the object
                - Prediction 3: Interior pixels of the object
                - Prediction 4: Image background

        :type output_transforms: numpy.array

        .. Example::
        
            >>> X_test = np.random.random((1,512,512,2))
            >>> output_image = app.predict(X_test, image_mpp=0.5, compartment='both)
            >>> print(output_image.shape)
                    (1, 512, 512, 2)

        """

        default_kwargs_cell = {
            'maxima_threshold': 0.075,
            'maxima_smooth': 0,
            'interior_threshold': 0.2,
            'interior_smooth': 2,
            'small_objects_threshold': 15,
            'fill_holes_threshold': 15,
            'radius': 2,
            'maxima_index': 4,
            'interior_index': 6
        }

        default_kwargs_nuc = {
            'maxima_threshold': 0.1,
            'maxima_smooth': 0,
            'interior_threshold': 0.2,
            'interior_smooth': 2,
            'small_objects_threshold': 15,
            'fill_holes_threshold': 15,
            'radius': 2,
            'maxima_index': 0,
            'interior_index': 2
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
        output_tiles = np.zeros((B_tiles,) + (8,) + self.model_image_shape)

        for tile_batch_start in range(0, tiles.shape[0], batch_size):
            # Load only this batch to GPU
            tile_batch = torch.tensor(tiles[tile_batch_start:tile_batch_start+batch_size]).to(self.device)
            
            with torch.inference_mode():
                pred = self.model(tile_batch)
                        
            # Move predictions back to CPU to save GPU memory
            output_tiles[tile_batch_start:tile_batch_start+batch_size] = pred.cpu()
            
            # Clean up GPU memory
            del tile_batch, pred
            if self.device != 'cpu':
                torch.cuda.empty_cache()


        # Untile images
        output_images = untile_output(output_tiles, tiles_info)

        label_image = mesmer_postprocess(
                                        output_images,
                                        compartment=compartment,
                                        whole_cell_kwargs=postprocess_kwargs_whole_cell,
                                        nuclear_kwargs = postprocess_kwargs_nuclear
                                        )

        label_image = resize_output(label_image, orig_img_shape)

        if return_transforms:
            return label_image, output_images
        else:
            return label_image
