import math
import torch
import torch.nn.functional as F
from torch import nn
from losses.gather import GatherLayer


class MixupLoss(nn.Module):
    """ Pairwise loss between before-mixup and after-mixup features.

    Reference:

    Args:

    Note that we use cosine similarity, rather than Euclidean distance in the original paper.
    """
    def __init__(self, sim='cos'):
        super().__init__()
        self.sim = sim

    def forward(self, inputs, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        # l2-normlize
        inputs = F.normalize(inputs, p=2, dim=1)
        n = inputs.size(0)

        # gather all samples from different GPUs as gallery to compute pairwise loss.
        gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)

        # compute distance between positive pairs: before-mixup and after-mixup
        inputs_ori = inputs[:n // 2]
        inputs_mixup = inputs[n // 2:]
        simvec = torch.sum(inputs_ori * inputs_mixup, dim=1)
        loss = torch.sum(1 - simvec)

        return loss