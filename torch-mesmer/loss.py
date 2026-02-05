import torch
from torch import nn
from torch.nn import functional as F

class SemanticLoss(nn.Module):

    def __init__(self, n_semantic_classes=[1,3,1,3], semantic_type=['cont','disc','cont','disc']):
        super().__init__()

        self.n_semantic_classes=n_semantic_classes
        self.semantic_type=semantic_type
        self.wcce = WCCE()
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):

        '''
        Semantic loss calculating losses for each of the semantic heads and summing together.
        Head with 1 class gets MSELoss (id and od) and head with more than one class gets WCCE (fgbg)

        Args:
            y_pred: List of Tensors representing predictions
            y_true: List of Tensors representing true values

        '''

        loss = 0
        counter = 0

        for n_heads in self.n_semantic_classes:

            curr_y_pred = y_pred[:,counter:counter+n_heads]
            curr_y_true = y_true[:,counter:counter+n_heads]

            if n_heads > 1:
                
                curr_y_pred = curr_y_pred.permute(0,2,3,1).flatten(0, -2)
                curr_y_true = curr_y_true.permute(0,2,3,1).flatten(0, -2)

                head_loss = F.cross_entropy(curr_y_pred, curr_y_true, ignore_index=-1)

                counter+=n_heads

            else:

                curr_y_pred = curr_y_pred.flatten()
                curr_y_true = curr_y_true.flatten()

                head_loss = F.mse_loss(curr_y_pred, curr_y_true, reduction='mean') * 0.01

                counter+=n_heads

            loss += head_loss

        return loss

        
class WCCE(nn.Module):

    def __init__(self, reduce=True):
        super().__init__()
        self.reduce = reduce

    def forward(self, y_pred, y_true, n_classes=3):
        
        eps = 1e-10
        _epsilon = torch.tensor(eps).type(y_pred.dtype).to(y_pred.device)

        y_pred = y_pred.reshape(-1, n_classes)
        y_true = y_true.reshape(-1, n_classes)

        print(torch.sum(y_pred, dim=0, keepdim=True).shape)

        y_pred = y_pred / torch.sum(y_pred, dim=0, keepdims=True)

        print(torch.sum(y_pred, axis=0, keepdim=True))
        
        # Clamp predictions to avoid log(0)
        y_pred = torch.clamp(y_pred, min=_epsilon, max=(1. - _epsilon))
            
        # Total number of valid (non-masked) samples across all classes
        total_sum = torch.sum(y_true)
        print(total_sum)
        
        # Sum per class across all samples (N dimension)
        class_sum = torch.sum(y_true, dim=0, keepdims=True)
        print(class_sum)
        
        # Compute weights: inverse frequency normalized by n_classes
        class_weights = 1.0 / n_classes * torch.divide(total_sum, class_sum + 1.)

        print(class_weights)

        class_weights = class_weights.to(y_pred.device)
        
        full_loss = - (y_true * torch.log(y_pred) * class_weights)
        print(full_loss.shape)
        if self.reduce:
            return full_loss.mean()
        else:
            return full_loss
        
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
            predictions: (B, T, N, M, 3) logits
            targets: (B, T, N, M, 3) one hot encoding of labels
        """

        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size
    
    def get_loss(self):
        """Compute and return current metrics."""
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
