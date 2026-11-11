import torch
from torch import nn

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
    Loss for the semantic heads of the panoptic network, as the published
    Mesmer recipe trains them.

    The loss of each head depends on its number of channels:

    * 1 channel (inner distance): mean squared error, weight 1.
    * more than 1 channel (pixelwise edge/interior/background): class-weighted
      categorical cross-entropy (see ``weighted_categorical_crossentropy``),
      multiplied by ``ce_weight`` (0.01 in the paper).

    The per-head losses are summed with no further weighting.

    Args:
        n_semantic_classes: the number of channels of each head, in output order.
        ce_weight: scale applied to the cross-entropy heads.
    '''

    def __init__(self, n_semantic_classes=(1, 3, 1, 3), ce_weight=0.01):
        super().__init__()

        self.n_semantic_classes = list(n_semantic_classes)
        self.ce_weight = ce_weight

    def forward(self, y_pred, y_true):

        '''
        Args:
            y_pred: model output, (B, sum(n_semantic_classes), H, W). The
                multi-channel heads must already be softmax probabilities.
            y_true: targets, same shape. Multi-channel heads are one-hot.

        Returns:
            Scalar tensor: the sum of the per-head losses.
        '''

        loss = 0
        start = 0

        for n_channels in self.n_semantic_classes:

            head_pred = y_pred[:, start:start + n_channels]
            head_true = y_true[:, start:start + n_channels]
            start += n_channels

            if n_channels > 1:
                head_loss = self.ce_weight * weighted_categorical_crossentropy(head_pred, head_true)
            else:
                head_loss = torch.mean((head_pred.float() - head_true.float()) ** 2)

            loss = loss + head_loss

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
