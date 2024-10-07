import torch
import torch.nn as nn

def weighted_categorical_crossentropy(y_true, y_pred,
                                      n_classes=3, axis=None, device=torch.device("cpu"),
                                      from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.
    Automatically computes the class weights from the target image and uses
    them to weight the cross entropy

    Args:
        y_true: A tensor of the same shape as ``y_pred``.
        y_pred: A tensor resulting from a softmax
            (unless ``from_logits`` is ``True``, in which
            case ``y_pred`` is expected to be the logits).
        from_logits: Boolean, whether ``y_pred`` is the
            result of a softmax, or is a tensor of logits.

    Returns:
        tensor: Output tensor.
    """
    if from_logits:
        raise Exception('weighted_categorical_crossentropy cannot take logits')
    y_true = torch.tensor(y_true).to(device)
    # y_true = K.cast(y_true, y_pred.dtype)
    # n_classes = K.cast(n_classes, y_pred.dtype)
    if axis is None:
        axis = 1 # if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1
    reduce_axis = [x for x in list(range(torch.Tensor.dim(y_pred))) if x != axis]
    # scale preds so that the class probas of each sample sum to 1
    y_pred = y_pred / torch.sum(y_pred, dim=axis, keepdims=True)
    # manual computation of crossentropy
    eps=1e-10
    _epsilon = torch.tensor(eps).type(y_pred.dtype).to(device)#.base_dtype)
    y_pred = torch.clamp(y_pred, min=_epsilon, max=(1. - _epsilon))
    total_sum = torch.sum(y_true)
    class_sum = torch.sum(y_true, dim=reduce_axis, keepdims=True)
    class_weights = 1.0 / n_classes * torch.divide(total_sum, class_sum + 1.)
    return - torch.sum((y_true * torch.log(y_pred) * class_weights), dim=axis)



def semantic_loss(n_classes, device):
    def _semantic_loss(y_pred, y_true):
        # assert(list(y_pred.size())==list(y_true.shape))
        if n_classes > 1:
            tmp_loss = 0.01 * weighted_categorical_crossentropy(y_true, y_pred, n_classes=n_classes, device=device)
            return torch.mean(tmp_loss)
        else:
            loss = nn.MSELoss()
            tmp_loss = loss(y_pred, torch.Tensor(y_true).to(device))
            return tmp_loss

    return _semantic_loss
