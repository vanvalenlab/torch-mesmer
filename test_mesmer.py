import numpy as np

from unittest.mock import Mock

import torch

from tensorflow.python.platform import test

# from deepcell.model_zoo import PanopticNet
from panoptic import PanopticNet
from mesmer import Mesmer
from deepcell.applications import MultiplexSegmentation

class TestMesmer(test.TestCase): 

    def test_mesmer_app(self, load_path=None):
        with self.cached_session():
            whole_cell_classes = [1, 3]
            nuclear_classes = [1, 3]
            num_semantic_classes = whole_cell_classes + nuclear_classes
            num_semantic_heads = len(num_semantic_classes)

            
            if load_path is not None:
                model = torch.load(load_path, weights_only=False)
            else:
                model = PanopticNet(
                    'resnet50',
                    input_shape=(256, 256, 2),
                    norm_method=None,
                    num_semantic_heads=num_semantic_heads,
                    num_semantic_classes=num_semantic_classes,
                    location=True,
                    include_top=True,
                    use_imagenet=False)

            model.eval()

            with torch.no_grad():
                app = Mesmer(model)
    
                # test output shape
                # shape = app.model.output_shape
                # self.assertIsInstance(shape, list)
                # self.assertEqual(len(shape), 4)
    
                # test predict with default
                x = np.random.rand(1, 500, 500, 2)
                # x = np.random.rand(1, 2, 500, 500)
                y = app.predict(x)
                self.assertEqual(x.shape[:-1], y.shape[:-1])
    
                # test predict with nuclear compartment only
                x = np.random.rand(1, 500, 500, 2)
                y = app.predict(x, compartment='nuclear')
                self.assertEqual(x.shape[:-1], y.shape[:-1])
                self.assertEqual(y.shape[-1], 1)
    
                # test predict with cell compartment only
                x = np.random.rand(1, 500, 500, 2)
                y = app.predict(x, compartment='whole-cell')
                self.assertEqual(x.shape[:-1], y.shape[:-1])
                self.assertEqual(y.shape[-1], 1)
    
                # test predict with both cell and nuclear compartments
                x = np.random.rand(1, 500, 500, 2)
                y = app.predict(x, compartment='both')
                self.assertEqual(x.shape[:-1], y.shape[:-1])
                self.assertEqual(y.shape[-1], 2)
    
                # test that kwargs are passed through successfully
                app._predict_segmentation = Mock()
    
                # get defaults
                _ = app.predict(x, compartment='whole-cell')
                args = app._predict_segmentation.call_args[1]
                default_cell_kwargs = args['postprocess_kwargs']['whole_cell_kwargs']
                default_nuc_kwargs = args['postprocess_kwargs']['nuclear_kwargs']
    
                # change one of the args for each compartment
                maxima_threshold_cell = default_cell_kwargs['maxima_threshold'] + 0.1
                radius_nuc = default_nuc_kwargs['radius'] + 2
    
                _ = app.predict(x, compartment='whole-cell',
                                postprocess_kwargs_whole_cell={'maxima_threshold':
                                                               maxima_threshold_cell},
                                postprocess_kwargs_nuclear={'radius': radius_nuc})
    
                args = app._predict_segmentation.call_args[1]
                cell_kwargs = args['postprocess_kwargs']['whole_cell_kwargs']
                assert cell_kwargs['maxima_threshold'] == maxima_threshold_cell
    
                nuc_kwargs = args['postprocess_kwargs']['nuclear_kwargs']
                assert nuc_kwargs['radius'] == radius_nuc
    
                # check that rest are unchanged
                cell_kwargs['maxima_threshold'] = default_cell_kwargs['maxima_threshold']
                assert cell_kwargs == default_cell_kwargs
    
                nuc_kwargs['radius'] = default_nuc_kwargs['radius']
                assert nuc_kwargs == default_nuc_kwargs
    
                # test legacy version
                old_app = MultiplexSegmentation(model)
    
                # test predict with default
                x = np.random.rand(1, 500, 500, 2)
                y = old_app.predict(x)
                self.assertEqual(x.shape[:-1], y.shape[:-1])

                torch.save(model, "/data/saved_model.pth")
    
                print("SUCCESS")

a = TestMesmer()
a.test_mesmer_app()
