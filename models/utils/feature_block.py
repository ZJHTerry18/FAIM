from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import random
import math
import numpy as np

class PatchShuffle:
    def __init__(self, patch_size=16, threshold=0.95):
        self.p = patch_size
        self.t = threshold  # clothes area(%) >= threshold, select as clothes patch
    
    def patch_embedding(self, features, upcloth_masks, pants_masks):
        '''
            convert features and masks to patches

            input:
                features: (B, C, H, W)
                masks: (B, mH, mW)

            output:
                feature_patches: (B, N_H, N_W, C, P*P)
                mask_patches: (B, N_H, N_W, mP*mP)
        '''
        b, c, h, w = features.shape
        mb, mh, mw = upcloth_masks.shape
        ratio = mh // h
        p = self.p         # patch size for feature map
        p_mask = p * ratio # patch size for mask

        h_n = h // p  # patch number on height dimension
        w_n = w // p  # patch number on width dimension
        feature_patches = features.reshape(b, c, h_n, p, w_n, p).permute(0,2,4,1,3,5).reshape(b, h_n, w_n, c, p*p)
        upcloth_mask_patches = upcloth_masks.reshape(mb, h_n, p_mask, w_n, p_mask).permute(0,1,3,2,4).reshape(mb, h_n, w_n, p_mask*p_mask)
        pants_mask_patches = pants_masks.reshape(mb, h_n, p_mask, w_n, p_mask).permute(0,1,3,2,4).reshape(mb, h_n, w_n, p_mask*p_mask)

        return feature_patches, upcloth_mask_patches, pants_mask_patches
    
    def re_patch_embedding(self, feature_patches):
        '''
            convert feature patches to original feature maps
            
            input:
                features_patches: (B, N_H, N_W, C, P*P)

            output:
                features: (B, C, H, W)
        '''
        b, h_n, w_n, c, _ = feature_patches.shape
        p = self.p
        features = feature_patches.reshape(b, h_n, w_n, c, p, p).permute(0,3,1,4,2,5).reshape(b, c, h_n*p, w_n*p)
        return features
    
    def __call__(self, features, masks, idx_shuff):
        '''
            shuffle patches that belong to clothes area in a batch

            input:
                features: feature map, (B,C,H,W)
                masks: human parsing mask, (B,mH,mW) 
                idx_shuff: list, shuffled indexes, ranging from 0 to B-1
            
            output:
                features_out: feature map after patch shuffling, (B,C,H,W)
        '''
        b, c, h, w = features.shape
        mb, mh, mw = masks.shape
        assert mh / h == mw / w and mh % h == 0 and mw % w == 0, 'mask size (%d, %d) and feature map size (%d, %d) unmatch' % (mh, mw, h, w)
        assert h % self.p == 0 and w % self.p == 0, 'feature size (%d, %d) cannot be divided by patch size (%d, %d)' % (h, w, self.p, self.p)

        # generate mask for upper clothes area
        cloth_masks = torch.where(masks == 2, torch.ones_like(masks), torch.zeros_like(masks)).to(float)
        pants_masks = torch.where(masks == 3, torch.ones_like(masks), torch.zeros_like(masks)).to(float)

        # patch_embedding
        feature_patches, upcloth_mask_patches, pants_mask_patches = self.patch_embedding(features, cloth_masks, pants_masks)
        old_feature_patches = feature_patches.clone()
        # print(feature_patches.shape, mask_patches.shape)

        # get cloth area patch indexes
        upcloth_mask_scores = torch.mean(upcloth_mask_patches, dim=3)
        upcloth_mask_scores[upcloth_mask_scores < self.t] = 0
        upcloth_indexes = [torch.argwhere(upcloth_mask_scores[i] > 0) for i in range(b)]  # i-th element is all the x-y position of cloth patches in i-th sample
        pants_mask_scores = torch.mean(pants_mask_patches, dim=3)
        pants_mask_scores[pants_mask_scores < self.t] = 0
        pants_indexes = [torch.argwhere(pants_mask_scores[i] > 0) for i in range(b)]  # i-th element is all the x-y position of cloth patches in i-th sample
        # print(upcloth_indexes[0], pants_indexes[0])


        # shuffle patches among samples: change cloth patches in src with cloth patches in dst
        for src_i in range(b):
            dst_i = idx_shuff[src_i]

            # for upper clothes
            src_num = upcloth_indexes[src_i].shape[0] # number of cloth patches in src
            dst_num = upcloth_indexes[dst_i].shape[0] # number of cloth patches in dst
            change_num = min(src_num, dst_num)      # number of patches to change
            # print('upcloth change num:', change_num)
            if change_num > 0:
                src_ind = random.sample(list(range(src_num)), change_num) # indexes to be changed in src
                dst_ind = random.sample(list(range(dst_num)), change_num) # indexes for changing in dst
                for si, di in zip(src_ind, dst_ind):
                    sx = upcloth_indexes[src_i][si][0]
                    sy = upcloth_indexes[src_i][si][1]
                    dx = upcloth_indexes[dst_i][di][0]
                    dy = upcloth_indexes[dst_i][di][1]
                    feature_patches[src_i][sx][sy] = old_feature_patches[dst_i][dx][dy]
            
            # for pants
            src_num = pants_indexes[src_i].shape[0] # number of cloth patches in src
            dst_num = pants_indexes[dst_i].shape[0] # number of cloth patches in dst
            change_num = min(src_num, dst_num)      # number of patches to change
            # print('pants change num:', change_num)
            if change_num > 0:
                src_ind = random.sample(list(range(src_num)), change_num) # indexes to be changed in src
                dst_ind = random.sample(list(range(dst_num)), change_num) # indexes for changing in dst
                for si, di in zip(src_ind, dst_ind):
                    sx = pants_indexes[src_i][si][0]
                    sy = pants_indexes[src_i][si][1]
                    dx = pants_indexes[dst_i][di][0]
                    dy = pants_indexes[dst_i][di][1]
                    feature_patches[src_i][sx][sy] = old_feature_patches[dst_i][dx][dy]
        
        # reverse patch embedding
        features_out = self.re_patch_embedding(feature_patches)

        return features_out

class FeatureClothErase(object):
    def __init__(self, p=0.1, erase_type='rce'):
        self.p = p
        self.erase_type = erase_type
    
    def __call__(self, features, masks):
        '''
            features: [B, C, H, W]
            masks: [B, H, W]
        '''
        B, C, H, W = features.shape
        _, mH, mW = masks.shape

        upcloth_mask_batch = torch.zeros_like(features).cuda()
        pants_mask_batch = torch.zeros_like(features).cuda()
        for i in range(B):
            mask_map = masks[i]

            upcloth_mask = torch.where(mask_map == 2, torch.ones_like(mask_map), torch.zeros_like(mask_map)).to(float)
            pants_mask = torch.where(mask_map == 3, torch.ones_like(mask_map), torch.zeros_like(mask_map)).to(float)
            mask_resize = torchvision.transforms.Resize([H, W])
            if self.erase_type == 'rce':
                if random.uniform(0, 1) < self.p:
                    upcloth_mask_batch[i] = mask_resize(upcloth_mask.unsqueeze(0)).to(int).sign().expand(C, H, W)
                if random.uniform(0, 1) < self.p:
                    pants_mask_batch[i] = mask_resize(pants_mask.unsqueeze(0)).to(int).sign().expand(C, H, W)
        features = features * (1 - upcloth_mask_batch)
        features = features * (1 - pants_mask_batch)
        # elif self.erase_type == 'rceb':
        #     if random.uniform(0, 1) < self.p:
        #         feature_map *= (1 - upcloth_mask)
        #         feature_map *= (1 - pants_mask)
        # elif self.erase_type == 'rcee':
        #     if random.uniform(0, 1) < self.p:
        #         if random.uniform(0, 1) < 0.5:
        #             feature_map *= (1 - upcloth_mask)
        #         else:
        #             feature_map *= (1 - pants_mask)
        
        return features

class FeatureClothErase2(nn.Module):
    def __init__(self, in_channels, p=0.1):
        super().__init__()
        self.p = p
        self.in_channels = in_channels

        self.conv_erase = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.in_channels)
        )

        for m in self.conv_erase:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal_(m.weight.data, 0.0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight.data, 0.0)
                init.constant_(m.bias.data, 0.0)

    def generate_mask(self, input_mask, output_h, output_w):
        B = input_mask.shape[0]
        output_mask = torch.ones(B, output_h, output_w).cuda()
        for i in range(B):
            if random.uniform(0, 1) < self.p:
                mask = torch.where((input_mask[i] == 2) + (input_mask[i] == 3), 
                                   torch.ones_like(input_mask[i]), torch.zeros_like(input_mask[i]))
                mask = mask.unsqueeze(0)
                mask_resize = torchvision.transforms.Resize([output_h, output_w])
                mask = mask_resize(mask).to(int).sign().to(torch.float32)
                output_mask[i] = 1.0 - mask

        return output_mask

    def forward(self, features, masks):
        '''
            randomly erase features with masks
            features: [B, C, H, W]
            masks: [B, mH, mW]
        '''
        B, C, H, W = features.shape

        masks = self.generate_mask(masks, H, W)
        masked_features = features * masks.unsqueeze(1)
        res = masked_features.view(B, C, -1).mean(-1)  # res:[B, C]
        res = self.conv_erase(res.view(B, C, 1, 1))
        outputs = features + res
        
        return outputs

class FeatureRenorm(nn.Module):
    def __init__(self,):
        super(FeatureRenorm, self).__init__()
    
    def forward(self, feature, mask, idx_shuff):
        b, c, h, w = feature.size()
        mask = ((mask==2) + (mask==3)).float().unsqueeze(1)
        mask = (nn.functional.interpolate(mask, size=(h, w), mode='nearest') == 1).expand(b, c, h, w)
        mean_i = feature.flatten(2).mean(dim=2)
        std_i = feature.flatten(2).std(dim=2)
        mean_c = torch.zeros(b, c).cuda()
        std_c = torch.ones(b, c).cuda()
        for i in range(b):
            if mask[i].sum() > c:
                mean_c[i] = feature[i].masked_select(mask[i]).view(c, -1).mean(dim=1) # [64,96,48] -> [64,]
                std_c[i] = feature[i].masked_select(mask[i]).view(c, -1).std(dim=1)
           
        norm_mean = mean_i.view(b, c, 1, 1)
        norm_std = std_i.view(b, c, 1, 1)
        denorm_mean = (0.5 * mean_i[idx_shuff] + 0.5 * mean_i).view(b, c, 1, 1)
        denorm_std = (0.5 * std_i[idx_shuff] + 0.5 * std_i).view(b, c, 1, 1)
        # norm_mean = mean_c.view(b, c, 1, 1).detach()
        # norm_std = std_c.view(b, c, 1, 1).detach()
        # denorm_mean = ((mean_c + mean_c[idx_shuff]) * 0.5).view(b, c, 1, 1).detach()
        # denorm_std = ((std_c + std_c[idx_shuff]) * 0.5).view(b, c, 1, 1).detach()

        feature_norm = (feature - norm_mean) / (norm_std + 1e-5)
        feature_aug = feature_norm * denorm_std + denorm_mean
        feature = feature_aug * mask.float() + feature * (1 - mask.float())

        return feature_aug

class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class ChannelAttn(nn.Module):
    """Channel Attention """
    def __init__(self, in_channels, is_softmax=False, reduction_rate=16, **kwargs):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels//reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels//reduction_rate, in_channels, 1)
        self.is_softmax = is_softmax

    def forward(self, x):
        x_in = x
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = F.relu(self.conv1(x))
        x = self.conv2(x) #[B,C,1,1]
        if self.is_softmax:
            x_att = F.softmax(x, 1)
        else:
            x_att = F.sigmoid(x)
        # z = x_in * x_att
        return x_att

class CBAMChannel(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAMChannel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        # maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout)


class CBAMSpatial(nn.Module):
    def __init__(self, kernel_size=7):
        super(CBAMSpatial, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


'''
x-->channel attention-->mixup-->add
'''
class MixupBlock(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(MixupBlock, self).__init__()
        self.channel_attn_id = ChannelAttn(in_channels)
        # self.channel_attn_cloth = ChannelAttn(in_channels, is_softmax=True)
    
    def forward(self, x, train, idx_shuff):
        '''x: tensor [B, C, H, W]'''
        w_id = self.channel_attn_id(x) # cloth-irrelevant feature: [B, C, H, W]
        # print(torch.std_mean(w_id))
        x_id = x * w_id
        x_cloth = x * (1 - w_id)
        # x_cloth = self.channel_attn_cloth(x) # cloth-relevant feature: [B, C, H, W]
        
        if idx_shuff is None:
            idx_shuff = list(range(x_cloth.shape[0]))
        if train:
            random.shuffle(idx_shuff) # shuffled index in [0,...,B-1] 

        x_out = x_id + x_cloth[idx_shuff] # recombined feature
        
        # use original feature and recombined feature, expanding batch size to 2*N
        if train:
            x_out = torch.cat((x, x_out), dim=0)

        return x_out


'''
x-->split-->(channel=C/2)-->mixup-->concat
'''
class MixupBlock2(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(MixupBlock2, self).__init__()
        self.in_channels = in_channels // 2
        # self.channel_attn = ChannelAttn(in_channels)
    
    def forward(self, x, train, idx_shuff):
        '''x: tensor [B, C, H, W]'''
        x_id = x[:,:self.in_channels, :, :]
        x_cloth = x[:, self.in_channels:, :, :]

        if idx_shuff is None:  
            idx_shuff = list(range(x_cloth.shape[0]))
        # if train:
        #     random.shuffle(idx_shuff) # shuffled index in [0,...,B-1]
        x_out = torch.cat((x_id, x_cloth[idx_shuff]), dim=1) # recombined feature
        # w_out = self.channel_attn(x_out)
        # x_out = w_out * x_out

        # use original feature and recombined feature, expanding batch size to 2*N
        if train:
            x_out = torch.cat((x, x_out), dim=0)
            
        return x_out, x_id, x_cloth
