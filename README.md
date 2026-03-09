# Torch Mesmer

A PyTorch implementation of the Mesmer pipeline for segmenting whole-slide tissue imaging. 

## The Network

Mesmer is built on a Panoptic network, consisting of a [Resnet50](https://arxiv.org/abs/1512.03385) backbone connected to a [feature pyramid network](https://arxiv.org/abs/1612.03144). The levels of the backbone and feature pyramid network can be selected, but for the pre-trained Mesmer model, we use backbone levels `C3-C5`, and pyramid levels `P3-P7`. Pyramid levels are then upsampled to match the input resolution (256 x 256 px) and delivered to the semantic heads of the model.

 In the pre-trained Mesmer model, there are four semantic heads. Heads 1 and 2 predict the whole cell transforms, and heads 3 and 4 predict the nuclear transforms.

```
Head 1 (1, 256, 256)
└─ Inner distance transform for whole cell

Head 2 (3, 256, 256)
├─ Boundary pixels for whole cell
├─ Interior pixels for whole cell
└─ Background pixels for whole cell

Head 3 (1, 256, 256)
└─ Inner distance transform for nucleus

Head 4 (3, 256, 256)
├─ Boundary pixels for nucleus
├─ Interior pixels for nucleus
└─ Background pixels for nucleus
```
After softmax on the semantic head convolutions, the model concatenates all predictions into an output tensor of shape `(8, 256, 256)` and returns it.

## The Dataset

Each dataset contains one nuclear and one cytosol channel, as well as the labeled ground truth mask for each channel, and metadata that contains the source tissue and experiment of each image.

- **Training**: 2576 square images (512 x 512)

- **Validation**: 3114 square images (256 x 256 px)

- **Test**: 1320 square images (256 x 256 px)

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

We use the Adam optimizer with a learning rate of 0.0001. Upon a plateau in validation loss that lasted longer than 5 epochs, the model's learning rate is reduced by one third.

The loss function is a combination of weighted categorical cross entropy (WCCE) and mean squared error (MSE) loss. For continuous predictions (inner distance transforms), MSE loss was used. For categorical predictions (foreground, background, perimeter), WCCE loss was used with class weights calculated for each batch. Loss from continuous heads was weighted with 0.01 to increase stability during training. The loss calculated from each head was summed and then used in backpropagation.

We used a batch size of 10 images, and an augmented version of each images was seen only once during each epoch. The model was trained for 16 epochs, and the model with the lowest validation loss was selected for testing.

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

 Each of these metrics was calculated for every image, allowing us to identify areas of weakness in each trained model.
