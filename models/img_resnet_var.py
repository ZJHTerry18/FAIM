import torchvision
import torch
from torch import nn
from torch.nn import init
import copy
import math
import random
from models.utils import pooling
from models.utils.feature_block import PatchShuffle, FeatureClothErase, FeatureClothErase2, FeatureRenorm
from models.utils.nonlocal_blocks import CADecouple
from models.utils.pdecouple import LayerScale_Block_CA, LayerScale_Block_CA_2
from models.utils.maskpooling import ClothMaskPooling
from IPython import embed

# ResNet50 with variance prediction
class ResNet50CA_var(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
        self.use_old_feature = config.MODEL.USE_OLD_FEATURE
        self.use_clo_var = config.MODEL.USE_CLO_VAR
        self.num_identities = kwargs['num_identities']
        self.num_clothes = kwargs['num_clothes']
        if self.use_clo_var:
            self.update_m = 0.1

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
        # self.base = nn.Sequential(*list(resnet50.children())[:-2])

        # resnet50 detailed parts
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        
        # main feature branch
        self.layer3_main = resnet50.layer3
        self.layer4_main = resnet50.layer4
        # decoupled branch
        self.layer3_dec = copy.deepcopy(self.layer3_main)
        self.layer4_id_feat = copy.deepcopy(self.layer4_main)
        self.layer4_id_var = copy.deepcopy(self.layer4_main)
        self.layer4_clo_feat = copy.deepcopy(self.layer4_main)
        # self.layer4_clo_var = copy.deepcopy(self.layer4_main)

        # self.var_head = FCN_head(in_dim=2048, mid_dim=[1024], out_dim=config.MODEL.CA_DIM)
        self.cross_clo_var = torch.zeros(1, config.MODEL.CA_DIM)
        self.cross_clo_var_memory = torch.zeros(self.num_identities, config.MODEL.CA_DIM)

        # self.intra_clo_var = torch.zeros(1, config.MODEL.CA_DIM)
        # self.intra_clo_var_memory = torch.zeros(self.num_clothes, config.MODEL.CA_DIM)

        self.patchshuffle_block = PatchShuffle(patch_size=8, threshold=0.95)

        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

        self.bn_main = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        self.bn_id = nn.BatchNorm1d(config.MODEL.CA_DIM)
        self.bn_cloth = nn.BatchNorm1d(config.MODEL.CA_DIM)
        init.normal_(self.bn_main.weight.data, 1.0, 0.02)
        init.constant_(self.bn_main.bias.data, 0.0)
        init.normal_(self.bn_id.weight.data, 1.0, 0.02)
        init.constant_(self.bn_id.bias.data, 0.0)
        init.normal_(self.bn_cloth.weight.data, 1.0, 0.02)
        init.constant_(self.bn_cloth.bias.data, 0.0)

    def forward(self, x, pids=None, clothes_ids=None, mask=None, train=False, idx_shuff=None, sample_k=-1):
        B, C, _, _ = x.size()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if train and self.do_patch_shuffle:
            x_out = self.patchshuffle_block(x, mask, idx_shuff=idx_shuff)
            if self.use_old_feature:
                x = torch.cat((x, x_out), dim=0)
            else:
                x = x_out

        x = self.layer1(x)
        base_x = self.layer2(x)

        # main branch
        x = self.layer3_main(base_x)
        x = self.layer4_main(x)

        f_main = self.globalpooling(x)
        f_main = f_main.view(f_main.size(0), -1)
        f_main = self.bn_main(f_main)
        
        # decouple branch
        x = self.layer3_dec(base_x)
        x_id = self.layer4_id_feat(x)
        x_cloth = self.layer4_clo_feat(x)

        f_id = self.globalpooling(x_id)
        f_id = f_id.view(f_id.size(0), -1)
        f_id = self.bn_id(f_id)

        f_cloth = self.globalpooling(x_cloth)
        f_cloth = f_cloth.view(f_cloth.size(0), -1)
        f_cloth = self.bn_cloth(f_cloth)

        var_id = self.layer4_id_var(x)
        var_id = (self.avgpooling(var_id) ** 2).view(var_id.size(0), -1).repeat_interleave(2, dim=1)
        gvar = var_id.mean(dim=1, keepdim=True)
        # var_id = self.var_head(var_id).mean(dim=3).mean(dim=2)
        # var_clo = self.layer4_clo_var(x)
        # var_clo = (self.avgpooling(var_clo) ** 2).view(var_clo.size(0), -1).repeat_interleave(2, dim=1)

        # feature augmentation by sampling from var_id: f_aug = f_id + e * var_id, where e is drawn from N(0, I)
        d = var_id.size(1)
        if train and sample_k > 0:
            if self.use_clo_var:
                mean_id_aug = f_id.repeat_interleave(sample_k * 2, dim=0)
                batch_cross_clo_var = calc_cross_clothes_variance(f_id.detach().cpu(), pids, clothes_ids)
                self.cross_clo_var = self.update_m * self.cross_clo_var + (1 - self.update_m) * batch_cross_clo_var.mean(dim=0, keepdim=True)
                # update_memory(self.cross_clo_var_memory, batch_cross_clo_var, pids, self.update_m)
                # cross_clo_var = self.cross_clo_var_memory[pids].cuda()
                # cross_clo_var = self.cross_clo_var.expand(B, -1).cuda()
                cross_clo_var = batch_cross_clo_var.cuda()
                # var_id = cross_clo_var * gvar_aug
                var_crossclo = cross_clo_var * gvar
                var_id_aug = var_id.unsqueeze(1).expand(-1, sample_k, -1)
                var_crossclo_aug = var_crossclo.unsqueeze(1).expand(-1, sample_k, -1)
                var_aug = torch.cat([var_id_aug, var_crossclo_aug], dim=1).reshape(-1, d)
            else:
                mean_id_aug = f_id.repeat_interleave(sample_k, dim=0)
                var_aug = var_id.repeat_interleave(sample_k, dim=0)
            epsilon = torch.randn(var_aug.size()).clamp(min=-3, max=3).cuda()
            f_id_aug = mean_id_aug + epsilon * var_aug

            # if self.use_clo_var:
            #     gvar_aug = var_clo.mean(dim=1, keepdim=True)
            #     batch_intra_clo_var = calc_intra_clothes_variance(f_cloth.detach().cpu(), pids, clothes_ids)
            #     self.intra_clo_var = self.update_m * self.intra_clo_var + (1 - self.update_m) * batch_intra_clo_var.mean(dim=0, keepdim=True)
            #     update_memory(self.intra_clo_var_memory, batch_intra_clo_var, clothes_ids, self.update_m)
            #     intra_clo_var = self.intra_clo_var_memory[clothes_ids].cuda()
            #     var_clo = intra_clo_var * gvar_aug
            # var_clo_aug = var_clo.repeat_interleave(sample_k, dim=0)
            # epsilon = torch.randn(var_id_aug.size()).clamp(min=-3, max=3).cuda()
            # f_clo_aug = mean_clo_aug + epsilon * var_clo_aug

            # return f_main, f_id, f_id_aug, f_cloth, var_id
            return f_main, f_id, f_id_aug, f_cloth, gvar

        return f_main, f_id, f_cloth, var_id

def calc_cross_clothes_variance(features, pids, clothes_ids):
    '''
        features: (B, c)
        pids: identity label, (B)
        clothes_ids: clothes label, (B)
    '''
    B, c = features.size()
    clo_mu = torch.zeros_like(features)
    for i in range(B):
        clo_mu[i] = features[clothes_ids == clothes_ids[i]].mean(dim=0)
    clo_var = torch.zeros_like(features)
    for i in range(B):
        mask = (pids == pids[i]) & (clothes_ids != clothes_ids[i])
        if mask.any():
            clo_var[i] = ((clo_mu[i] - features[mask]) ** 2).mean(dim=0)
    clo_var = nn.functional.normalize(clo_var, p=2, dim=1) * math.sqrt(c)
    return clo_var

def calc_intra_clothes_variance(features, pids, clothes_ids):
    '''
        features: (B, c)
        pids: identity label, (B)
        clothes_ids: clothes label, (B)
    '''
    B, c = features.size()
    clo_mu = torch.zeros_like(features)
    for i in range(B):
        clo_mu[i] = features[clothes_ids == clothes_ids[i]].mean(dim=0)
    clo_var = torch.zeros_like(features)
    for i in range(B):
        mask = (pids == pids[i]) & (clothes_ids == clothes_ids[i])
        if mask.any():
            clo_var[i] = ((clo_mu[i] - features[mask]) ** 2).mean(dim=0)
    clo_var = nn.functional.normalize(clo_var, p=2, dim=1) * math.sqrt(c)
    return clo_var

def update_memory(memory_bank, batch_f, pids, momentum=0.1):
    B, c = batch_f.size()
    for i in range(B):
        old_memory = memory_bank[pids[i]]
        memory_bank[pids[i]] = momentum * batch_f[i] + (1 - momentum) * old_memory
    

class FCN_head(nn.Module):
    def __init__(self, in_dim=1920, mid_dim=[256], out_dim=1024):
        super(FCN_head, self).__init__()
        BN_MOMENTUM = 0.1
        dim_list = [in_dim] + mid_dim + [out_dim]
        self.layer = nn.Sequential()
        self.bayes_count = len(dim_list) - 1
        for i in range(self.bayes_count):
            in_dim, out_dim = dim_list[i], dim_list[i + 1]
            self.layer.add_module('Conv_{}'.format(i),
                                  nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1,
                                            stride=1, padding=0))
            self.layer.add_module('BN_{}'.format(i), nn.BatchNorm2d(out_dim, momentum=BN_MOMENTUM))
            if i < self.bayes_count - 1:
                self.layer.add_module('ReLU_{}'.format(i), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)