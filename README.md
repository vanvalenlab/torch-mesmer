# Torch Mesmer

A PyTorch implementation of the Mesmer pipeline for segmenting whole-slide tissue imaging.

## The Network

Mesmer is built on a Panoptic network, consisting of a [Resnet50](https://arxiv.org/abs/1512.03385) backbone connected to a [feature pyramid network](https://arxiv.org/abs/1612.03144). The current model uses backbone levels `C3-C5`. As in the released TF Mesmer, each semantic head reads only the finest pyramid level, `P3`, and upsamples it to the input resolution (256 x 256 px); see "Weights and architecture versions" below.

In the pre-trained Mesmer model, there are four semantic heads. **Heads 1 and 2 predict the nuclear transforms, and heads 3 and 4 predict the whole-cell transforms.** This is the reverse of the original deepcell Mesmer, which puts whole-cell first.

```markdown
Head 1 (1, 256, 256)          -> output channel 0
└─ Inner distance transform for nucleus

Head 2 (3, 256, 256)          -> output channels 1-3
├─ Boundary pixels for nucleus
├─ Interior pixels for nucleus
└─ Background pixels for nucleus

Head 3 (1, 256, 256)          -> output channel 4
└─ Inner distance transform for whole cell

Head 4 (3, 256, 256)          -> output channels 5-7
├─ Boundary pixels for whole cell
├─ Interior pixels for whole cell
└─ Background pixels for whole cell
```

After softmax on the semantic head convolutions, the model concatenates all predictions into an output tensor of shape `(8, 256, 256)` and returns it.

## Using the model

```python
import numpy as np
from torch_mesmer.mesmer import Mesmer

app = Mesmer()  # uses a GPU if one is available (or pass device='cpu'); downloads the released weights if needed

# Channels first: (batch, 2, height, width), nuclear image in channel 0,
# membrane/cytoplasm image in channel 1.
image = np.random.random((1, 2, 512, 512)).astype(np.float32)

masks = app.predict(image, image_mpp=0.5, compartment='both')
masks.shape  # (1, 2, 512, 512): channel 0 nuclear mask, channel 1 whole-cell mask
```

- **Channel order is nuclear first, for both inputs and outputs.** With `compartment='whole-cell'` or `'nuclear'` the
  output has a single channel.
- **Inputs must be channels first.** For a channels-last `(batch, height, width, 2)` array, use
  `np.moveaxis(image, -1, 1)`; `predict` raises a `ValueError` otherwise.
- `image_mpp` is the image resolution in microns per pixel. The model works at 0.5; other values are resized to it and
  the masks are resized back. It is required.
- Images of any size are accepted: smaller images are padded to 256×256 and larger ones are tiled.
- `preprocess_kwargs`, `postprocess_kwargs_nuclear` and `postprocess_kwargs_whole_cell` override the defaults. The
  post-processing defaults are `DEFAULT_POSTPROCESS_KWARGS_NUCLEAR` / `_WHOLE_CELL` in `torch_mesmer/mesmer.py`.
- **`Mesmer()` with no `model_path` loads the retrained, benchmarked weights** (`torch-mesmer_2026-11-10.pth`),
  downloading them from Hugging Face into `~/.deepcell/models` the first time (see "Getting the weights" below). The
  post-processing defaults were tuned for these weights. To use the older `torch-mesmer_2026-06-30.pth`, download it
  from the same repository and pass its path as `model_path`.
- `torch_mesmer` no longer sets the number of CPU threads when imported. On a shared machine, call
  `torch.set_num_threads(n)` yourself.

### Getting the weights

The weights are on Hugging Face at [vanvalenlab/torch-mesmer](https://huggingface.co/vanvalenlab/torch-mesmer). They
are free for non-commercial use, but you need a Hugging Face account to download them:

1. Create a free [Hugging Face](https://huggingface.co) account, open the
   [model page](https://huggingface.co/vanvalenlab/torch-mesmer) and accept the license terms.
2. Create an access token on the [Access Tokens](https://huggingface.co/settings/tokens) page and export it as the
   environment variable `HF_TOKEN` (or log in once with `hf auth login`).

`Mesmer()` then downloads the weights the first time it runs, and reuses the downloaded file after that.

### Weights and architecture versions

The network changed in November 2026: the feature-pyramid wiring was fixed and Keras-style
BatchNorm and initialisation were adopted, and then the semantic heads were changed back to Mesmer's own design (each head
reads only the finest pyramid level, P3, through three linear 64-filter conv + 2x upsampling rounds; about 26M
parameters, as in the released TF Mesmer). **Weights trained before that change,
including the released `torch-mesmer_2026-06-30.pth`, are not compatible with the new network**, and new weights do
not fit the old one.

- Every checkpoint saved from the new network carries an `arch_version` entry (currently 3). Older files have none
  and count as version 1, which uses the earlier fused multi-level head. Version 2 (the pyramid fixes with the
  fused head) was never released and is no longer built.
- `Mesmer()` reads the checkpoint first and builds the matching network, so the 2026-06-30 weights keep giving exactly
  the masks they always did.
- Building the network yourself: use `PanopticNet(legacy=True)` for version 1 weights. Loading weights into the
  wrong version raises a `ValueError` that says which one to use. `legacy=True` is only for loading old weights;
  don't train with it.

## The Dataset

Each dataset contains one nuclear and one cytosol channel, as well as the labeled ground truth mask for each channel, and metadata that contains the source tissue and experiment of each image.

The data is TissueNet v1.1, converted to Zarr by `torch_mesmer/preprocess.py`:

- **Training**: 2580 square images (512 x 512 px)

- **Validation**: 3118 square images (256 x 256 px)

- **Test**: 1320 square images (256 x 256 px). The original file has 1324; the first 4 (the only spleen images) are left
  out because their nuclear ground truth does not match the nuclei.

**Note**: The training dataset is (512 x 512). This is so we can add data agumentation during training, which includes random rotations, flips, crops, and zooms. Validation and testing does not use data augmentation, so they are cropped to (256 x 256 px).

## The Training

### Data loaders

The training and validation data are loaded into a PyTorch `Dataset` object, which conducts preprocessing under the hood for each batch. This `Dataset` is then used in the construction of a `Dataloder`. The preprocessing and augmentation pipeline is outlined below:

1. Each item (one image and ground truth mask) is selected from the full dataset.
2. The image is normalized with two steps:
    1. Images are thresholded in order to reduce the influence of bright pixels.
    2. Imaes are then normalized using Contrast Limited Adaptive Histogram Equalization (CLAHE) with the `equalize_adapthist` function from [scikit-image](https://scikit-image.org/).
3. The labels are then transformed to generate the inner distance transform, the perimeter pixel mask, the interior pixel mask, and the background pixel mask for both whole cell and nuclear channels.
4. The normalized images and mask transformations are then augmented using a random combination of rotations, flips, crops and zooms.
5. The images and masks are then returned to the model.

### Optimizer, Learning Rate, and Loss Function

This follows the published Mesmer training recipe. We use the Adam optimizer with a learning rate of 0.0001, multiplied by 0.99 after every epoch. There is no plateau-based reduction and no early stopping.

The loss function is a combination of weighted categorical cross entropy (WCCE) and mean squared error (MSE) loss. For continuous predictions (inner distance transforms), MSE loss was used. For categorical predictions (foreground, background, perimeter), WCCE loss was used with class weights calculated for each batch. **Loss from the categorical (WCCE) heads was weighted with 0.01**, as in the paper. The loss calculated from each head was summed and then used in backpropagation.

We used a batch size of 8 images, and an augmented version of each images was seen only once during each epoch. The model was trained for 100 epochs, and we test the model that returned the lowest validation loss.

## The Testing

The model was used to segment 1320 test images. These segmentations were then compared to the ground truth using a custom metrics pipeline that analyzes the following:

- **Recall**
- **Precision**
- **Jaccard index (IoU)** - The index of overlap between the ground truth and the prediction
- **Gained detections** - objects segmented but not present in the ground truth
- **Missed detections** - objects present in the ground truth that were missed by the model
- **Splits** - number of "one to many" errors
- **Merges** - number of "many to one" errors
- **Catastrophes** - number of "many to many" errors

Each of these metrics is calculated for every image, allowing us to identify areas of weakness in each trained model. Recall, precision and F1 are then **pooled** over all images (counts summed before dividing), as in the Mesmer paper. Run it with `python -m torch_mesmer.eval --model-path <weights>`.

### Results

On the TissueNet v1.1 test split (1,320 images), the retrained port and the released TensorFlow Mesmer, run on the
same images and scored with the same code (2026-11-11):

| | Whole-cell F1 | Nuclear F1 |
|---|---|---|
| **torch-mesmer** (retrained weights, current defaults) | **0.826** | **0.893** |
| Released TF Mesmer (MultiplexSegmentation-9) | 0.811 | 0.878 |
| Mesmer paper (whole-cell; 2020 test split, not the same images) | 0.82 | — |

The port is ahead by 0.015 in both compartments (paired 95% interval about ±0.003). The paper's number was measured
on a different split and is a reference point, not a like-for-like comparison.
