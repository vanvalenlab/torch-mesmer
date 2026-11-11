import torch
from torch import nn
from torch.nn import functional as F

# Keras's K.epsilon(), used by deepcell's weighted_categorical_crossentropy.
KERAS_EPSILON = 1e-7


def weighted_categorical_crossentropy(y_pred, y_true, eps=KERAS_EPSILON):
    '''
    Class-weighted categorical cross-entropy on *probabilities*.

    Port of ``deepcell-tf/deepcell/losses.py:67-74``. The model's pixelwise heads
    already end in a softmax, so ``y_pred`` is treated as probabilities (no
    second softmax): it is renormalised over the class axis, clamped to
    ``[eps, 1 - eps]``, and the loss is ``-sum_c y * log(p) * w_c``, averaged
    over batch and pixels.

    The class weights are recomputed for every batch from ``y_true``:
    ``w_c = total / (n_classes * (class_sum_c + 1))``, where ``total`` is the
    number of labelled pixels in the batch. The ``+ 1`` keeps the weight finite
    when a class is absent from the batch.

    Args:
        y_pred: (B, C, H, W) class probabilities.
        y_true: (B, C, H, W) one-hot targets.
        eps: clamp value for the probabilities.

    Returns:
        Scalar tensor.
    '''
    # Compute in float32 even under mixed precision: eps=1e-7 underflows in fp16.
    y_pred = y_pred.float()
    y_true = y_true.float()

    n_classes = y_pred.shape[1]
    reduce_dims = [d for d in range(y_true.ndim) if d != 1]

    y_pred = y_pred / y_pred.sum(dim=1, keepdim=True)
    y_pred = y_pred.clamp(eps, 1.0 - eps)

    class_sum = y_true.sum(dim=reduce_dims, keepdim=True)
    class_weights = y_true.sum() / (n_classes * (class_sum + 1.0))

    per_pixel = -(y_true * torch.log(y_pred) * class_weights).sum(dim=1)
    return per_pixel.mean()


class SemanticLoss(nn.Module):

    '''
    Loss function for computing loss for each semantic head of the panoptic network.
    Loss is calculated depending on the number of channels each head has:
    1 Channel: output is continuous, so loss is calculated using MSE
    >1 Channel: output is classification, so we use an unweighted categorical cross-entropy
    Args:
        n_semantic_classes: the number of semantic classes that is returned for each head.
    '''

    def __init__(self, n_semantic_classes=[1,3,1,3], loss_weight = 0.01):
        super().__init__()

        self.n_semantic_classes=n_semantic_classes
        self.loss_weight = loss_weight

    def forward(self, y_pred, y_true):

        '''
        Semantic loss calculating losses for each of the semantic heads and summing together.
        Heads with 1 channel gets MSE loss, head with more than one channel are categorical, so we use CCE loss

        Args:
            y_pred: Tensors representing predictions (B, sum(n_semantic_classes), H, W)
            y_true: Tensors representing true values (B, sum(n_semantic_classes), H, W)

        '''

        loss = 0

        # counter for keeping track of indexing
        counter = 0

        for n_heads in self.n_semantic_classes:

            curr_y_pred = y_pred[:,counter:counter+n_heads]
            curr_y_true = y_true[:,counter:counter+n_heads]

            if n_heads > 1:

                # The head already ends in a softmax: y_pred holds probabilities
                # and y_true is one-hot.
                head_loss = weighted_categorical_crossentropy(curr_y_pred, curr_y_true)

                counter+=n_heads

            else:

                curr_y_pred = curr_y_pred.flatten()
                curr_y_true = curr_y_true.flatten()

                # reduce loss calculation for these heads for stable learning
                head_loss = F.mse_loss(curr_y_pred, curr_y_true, reduction='mean') * self.loss_weight

                counter+=n_heads

            loss += head_loss

        return loss

        
class LossTracker:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0

    def update(self, loss, batch_size):
        """
        Args:
            loss: scalar loss value
            batch_size: batch size
        """

        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size
    
    def get_loss(self):
        """Compute and return current loss."""
        avg_loss = self.total_loss / max(self.total_samples, 1)

        return avg_loss 

if __name__ == '__main__':

    n_semantic_classes = [1, 3, 1, 3]

    test_true = torch.rand(8, 8, 256, 256) > 0.5
    test_true = test_true.float()

    test_pred = torch.rand(8, 8, 256, 256) > 0.5
    test_pred = test_pred.float()
    
    test_mask = test_true[:,3].bool()

    loss = SemanticLoss(n_semantic_classes=n_semantic_classes)

    loss_out = loss(test_pred, test_true)

    print(loss_out)
