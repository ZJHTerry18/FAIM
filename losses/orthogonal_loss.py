import math
import torch
import torch.nn.functional as F
from torch import nn
from losses.gather import GatherLayer

class OrthogonalLoss(nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, features):
        '''
            input:
                features: [B, N, C], N is the number of features
            
            output:
                orthogonal loss. Aim to make the N features each orthogonal
        '''
        n_f = features.size(1)
        features = F.normalize(features, p=2, dim=2)
        i_mat = torch.eye(n_f).cuda()

        ortho_loss = torch.norm(torch.matmul(features, features.transpose(1,2).contiguous()) - i_mat, p='fro', dim=(1,2))
        ortho_loss = torch.mean(ortho_loss)

        return ortho_loss