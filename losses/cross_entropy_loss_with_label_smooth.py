import torch
from torch import nn


class CrossEntropyWithLabelSmooth(nn.Module):
    """ Cross entropy loss with label smoothing regularization.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. In CVPR, 2016.
    Equation: 
        y = (1 - epsilon) * y + epsilon / K.

    Args:
        epsilon (float): a hyper-parameter in the above equation.
    """
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        _, num_classes = inputs.size()
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (- targets * log_probs).mean(0).sum()

        return loss

class CrossEntropyWithClothes(nn.Module):
    """ Clothes-based Cross Entropy Loss.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
    """
    def __init__(self, epsilon=0.1, alpha=0.3):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, inputs, targets, positive_mask):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes). The clothes classes with 
                the same identity as the anchor sample are defined as positive clothes classes and their mask 
                values are 1. The clothes classes with different identities from the anchor sample are defined 
                as negative clothes classes and their mask values in positive_mask are 0.
        """
        identity_mask = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()
        positive_mask = torch.sign(identity_mask + positive_mask)
        negtive_mask = 1 - positive_mask

        exp_logits = torch.exp(inputs)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = inputs - log_sum_exp_pos_and_all_neg

        mask = (1 - self.epsilon) * identity_mask + (1 - self.epsilon) * self.alpha * positive_mask + \
            self.epsilon / negtive_mask.sum(1, keepdim=True) * negtive_mask
        loss = (- mask * log_prob).sum(1).mean()

        return loss
