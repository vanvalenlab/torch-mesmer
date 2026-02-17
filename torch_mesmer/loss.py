import torch
from torch import nn
from torch.nn import functional as F

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

                # y_pred needs to be softmaxed for this type of CCE calculation
                # y_true needs to be one-hot encoded
                
                curr_y_pred = curr_y_pred.permute(0,2,3,1).flatten(0, -2)
                curr_y_true = curr_y_true.permute(0,2,3,1).flatten(0, -2)

                class_sum = curr_y_true.sum(dim=0)
                weights = class_sum.sum()/(class_sum.shape[0] * class_sum)

                # loader returns padded values as -1, so set this as ignore_index
                head_loss = F.cross_entropy(curr_y_pred, curr_y_true, weight=weights, ignore_index=-1)

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
