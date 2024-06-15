import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import pooling

class ClothMaskPooling(nn.Module):
    def __init__(self, ptype='maxavg', t=0.8):
        super().__init__()
        self.ptype = ptype
        self.t = t

        if ptype == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool1d(1)
        elif ptype == 'max':
            self.globalpooling = nn.AdaptiveMaxPool1d(1)
        elif ptype == 'gem':
            self.globalpooling = pooling.GeMPooling(p=3)
        elif ptype == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling1d()
    
    def forward(self, x, mask):
        B, C, H, W = x.shape
        x_split = x.chunk(2, dim=1)
        mask = torch.cat((mask, mask), dim=0)
        _, mH, mW = mask.shape
        assert H / mH == W / mW
        patch_size = mH // H
        feat_c = C // 2 if self.ptype != 'maxavg' else C

        cloth_feat = torch.zeros((B, feat_c)).cuda()
        id_feat = torch.zeros((B, feat_c)).cuda()
        for bi in range(B):
            fmask = torch.zeros((H, W), dtype=bool).cuda()
            clothmask = torch.where(mask[bi] == 2, torch.ones_like(mask[bi]), torch.zeros_like(mask[bi])).to(float)
            pantmask = torch.where(mask[bi] == 3, torch.ones_like(mask[bi]), torch.zeros_like(mask[bi])).to(float)
            imgmask = torch.sign(clothmask + pantmask)
            for h in range(H):
                for w  in range(W):
                    m = torch.mean(imgmask[patch_size * h:patch_size * (h + 1), patch_size * w:patch_size * (w + 1)])
                    if m >= self.t:
                        fmask[h, w] = True
            cloth_num = torch.sum(fmask.to(int))

            idf = torch.masked_select(x_split[0][bi], ~fmask).reshape(C // 2, H*W - cloth_num).unsqueeze(0)
            idf = self.globalpooling(idf).squeeze()
            if cloth_num == 0:
                clothf = torch.masked_select(x_split[1][bi], ~fmask).reshape(C // 2, H * W).unsqueeze(0)
            else:
                clothf = torch.masked_select(x_split[1][bi], fmask).reshape(C // 2, cloth_num).unsqueeze(0)
            clothf = self.globalpooling(clothf).squeeze()
            id_feat[bi] = idf
            cloth_feat[bi] = clothf

        return id_feat, cloth_feat