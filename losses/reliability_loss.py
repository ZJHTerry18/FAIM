import math
import torch
import torch.nn.functional as F
from torch import nn

class ReliabilityCELoss(nn.Module):
    def __init__(self, lamda_aug=1.0, lamda_fu=0.3, gamma=1.0, epsilon=0.1):
        super().__init__()
        self.lamda_aug = lamda_aug
        self.lamda_fu = lamda_fu
        self.gamma = gamma
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.rankingloss = nn.MarginRankingLoss(margin=gamma)
    
    def forward(self, cls_outputs, targets, var, cls_outputs_aug=None, targets_aug=None):
        """
        Args:
            cls_outputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            var: variance vector of the mini-batch with shape (batch_size, feat_dim)
            cls_outputs: prediction matrix of augmented samples (before softmax) with shape (batch_size * aug_times, num_classes)
            targets_aug: ground truth labels of augmented samples with shape (batch_size * aug_times)
        """

        # g_var = var.mean(dim=1)
        g_var = var.squeeze()
        # feature uncertainty loss
        var_dim = var.size(1)
        fu_value = g_var #+ 0.5 * var_dim * (math.log(2 * math.pi) + 1)
        fu_loss = self.rankingloss(fu_value, torch.zeros_like(fu_value), torch.ones_like(fu_value))

        # cross-entropy loss of original samples
        _, num_classes = cls_outputs.size()
        log_probs = self.logsoftmax(cls_outputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss_ce = (- targets * log_probs).sum(dim=1) #/ (g_var + 1e-10)
        
        # reliability_loss = loss_ce / (var + 1e-10) + self.lamda * torch.log(var + 1e-10)
        if cls_outputs_aug is None:
            rel_loss = loss_ce.mean() + fu_loss
            return rel_loss
        
        # cross-entropy loss of augmented samples
        aug_k = int(cls_outputs_aug.size(0) / cls_outputs.size(0))
        _, num_classes = cls_outputs_aug.size()
        log_probs_aug = self.logsoftmax(cls_outputs_aug)
        targets_aug = torch.zeros(log_probs_aug.size()).scatter_(1, targets_aug.unsqueeze(1).data.cpu(), 1).cuda()
        targets_aug = (1 - self.epsilon) * targets_aug + self.epsilon / num_classes
        loss_ce_aug = (- targets_aug * log_probs_aug).sum(dim=1) #/ (g_var.repeat_interleave(aug_k) + 1e-10)

        # print('loss_ce: ', loss_ce.mean())
        # print('loss_ce_aug: ', loss_ce_aug.mean())
        # print('fu_loss: ', fu_loss)
        # print('fu value: ', fu_value)
        rel_loss = loss_ce.mean() + self.lamda_aug * loss_ce_aug.mean() + self.lamda_fu * fu_loss

        return rel_loss
