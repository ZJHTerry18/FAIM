import copy
import math
import torchvision
import torch
from torch import nn
from torch.nn import init
from models.utils import pooling
from models.utils.feature_block import MixupBlock, MixupBlock2, PatchShuffle
from models.utils.nonlocal_blocks import CADecouple
from models.swin_transformer_ import swin_t, swin_s, swin_b
from models.utils.pdecouple import LayerScale_Block_CA, LayerScale_Block_CA_2
from models.utils.maskpooling import ClothMaskPooling

class SwinT(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
        self.use_old_feature = config.MODEL.USE_OLD_FEATURE

        swint = swin_b(weights='IMAGENET1K_V1')
        
        self.patch_partition = swint.features[0]
        self.transformer_blocks = swint.features[1:]
        self.norm = swint.norm

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

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)


    def forward(self, x, mask=None, train=False, idx_shuff=None):
        x = self.patch_partition(x)

        # patch shuffle
        if train and self.do_patch_shuffle:
            x_p = x.permute(0, 3, 1, 2).contiguous()
            x_p = self.patchshuffle_block(x_p, mask, idx_shuff=idx_shuff)
            x_p = x_p.permute(0, 2, 3, 1).contiguous()
            if self.use_old_feature:
                x = torch.cat((x, x_p), dim=0)
            else:
                x = x_p

        x = self.transformer_blocks(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.globalpooling(x)
        x = torch.flatten(x, 1)
        x = self.bn(x)

        return x


class SwinTCA(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
        self.use_old_feature = config.MODEL.USE_OLD_FEATURE
        self.use_sim = config.MODEL.SIM
        self.num_decouple_blks = 2

        swint = swin_b(weights='IMAGENET1K_V1')

        self.cls_token_list = []
        for i in range(2):
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.MODEL.CA_DIM))
            self.cls_token_list.append(self.cls_token)
        
        self.patchshuffle_block = PatchShuffle(patch_size=8, threshold=0.95)
        if self.use_sim:
            self.maskpooling = ClothMaskPooling(ptype=config.MODEL.POOLING.NAME)
        
        self.patch_partition = swint.features[0]
        self.transformer_blocks = swint.features[1:-2]
        self.layer3 = swint.features[-2]
        self.layer4 = swint.features[-1]
        self.layer3_2 = copy.deepcopy(self.layer3)
        self.layer4_2 = copy.deepcopy(self.layer4)
        self.norm = swint.norm
        self.norm_2 = copy.deepcopy(self.norm)

        
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

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

        # cross-attention block
        self.ca_block = CADecouple(in_channels=config.MODEL.CA_DIM, out_channels=config.MODEL.CA_DIM,
            num_blocks=1, num_heads=1, inter_channels=config.MODEL.CA_DIM // 2)


    def forward(self, x, mask=None, train=False, idx_shuff=None):
        x = self.patch_partition(x)
        
        # patch shuffle
        if train and self.do_patch_shuffle:
            x_p = x.permute(0, 3, 1, 2).contiguous()
            x_p = self.patchshuffle_block(x_p, mask, idx_shuff=idx_shuff)
            x_p = x_p.permute(0, 2, 3, 1).contiguous()
            if self.use_old_feature:
                x = torch.cat((x, x_p), dim=0)
            else:
                x = x_p

        x = self.transformer_blocks(x)
        global_feat = self.layer3(x)
        global_feat = self.layer4(global_feat)
        global_feat = self.norm(global_feat)
        local_feat = self.layer3_2(x)
        local_feat = self.layer4_2(local_feat)
        local_feat = self.norm_2(local_feat)
        local_feat_chunk = local_feat.chunk(2, dim=2)

        x = x.permute(0, 3, 1, 2).contiguous()

        f_id, f_cloth = self.ca_block(x)
        f = self.globalpooling(x)
        f = torch.flatten(f, 1)
        f = self.bn(f)

        return f, f_id, f_cloth


class SwinTCA_2(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
        self.use_old_feature = config.MODEL.USE_OLD_FEATURE
        self.use_sim = config.MODEL.SIM
        self.num_decouple_blks = 2

        swint = swin_b(weights='IMAGENET1K_V1')

        self.cls_token_list = []
        for i in range(2):
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.MODEL.CA_DIM))
            self.cls_token_list.append(self.cls_token)
        
        self.patchshuffle_block = PatchShuffle(patch_size=8, threshold=0.95)
        if self.use_sim:
            self.maskpooling = ClothMaskPooling(ptype=config.MODEL.POOLING.NAME)
        
        # decoupling blocks
        self.decouple_blks = nn.ModuleList([
            LayerScale_Block_CA(
                dim=config.MODEL.FEATURE_DIM, num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                num_stripes=2)
            for _ in range(self.num_decouple_blks)])
  
        self.patch_partition = swint.features[0]
        self.transformer_blocks = swint.features[1:-2]
        self.layer3 = swint.features[-2]
        self.layer4 = swint.features[-1]
        # self.layer3_2 = copy.deepcopy(self.layer3)
        # self.layer4_2 = copy.deepcopy(self.layer4)
        self.norm = swint.norm
        # self.norm_2 = copy.deepcopy(self.norm)

        
        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool1d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool1d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling1d()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM * 2)
        self.bn_id = nn.BatchNorm1d(config.MODEL.CA_DIM)
        self.bn_cloth = nn.BatchNorm1d(config.MODEL.CA_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        init.normal_(self.bn_id.weight.data, 1.0, 0.02)
        init.constant_(self.bn_id.bias.data, 0.0)
        init.normal_(self.bn_cloth.weight.data, 1.0, 0.02)
        init.constant_(self.bn_cloth.bias.data, 0.0)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None, train=False, idx_shuff=None):
        x = self.patch_partition(x)
        
        # patch shuffle
        if train and self.do_patch_shuffle:
            x_p = x.permute(0, 3, 1, 2).contiguous()
            x_p = self.patchshuffle_block(x_p, mask, idx_shuff=idx_shuff)
            x_p = x_p.permute(0, 2, 3, 1).contiguous()
            if self.use_old_feature:
                x = torch.cat((x, x_p), dim=0)
            else:
                x = x_p

        x = self.transformer_blocks(x)

        B = x.size(0)
        global_feat = self.layer3(x)
        global_feat = self.layer4(global_feat)
        global_feat = self.norm(global_feat)
        feat_dim = global_feat.size(-1)
        # local_feat = self.layer3_2(x)
        # local_feat = self.layer4_2(local_feat)
        # local_feat = self.norm_2(local_feat)

        local_feat = global_feat.view(B, -1, feat_dim).contiguous()   # [b, h*w, c]
        global_feat = global_feat.permute(0, 3, 1, 2).contiguous()  # [b, c, h, w]
        local_feat_chunk = local_feat.chunk(2, dim=2)

        # decoupling attention
        cls_tokens = []
        for i in range(len(self.cls_token_list)):
            cls_token_tmp = self.cls_token_list[i].expand(B, -1, -1).cuda()
            cls_tokens.append(cls_token_tmp)
        
        for i, blk in enumerate(self.decouple_blks):
            cls_tokens = blk(global_feat, local_feat_chunk, cls_tokens)
        if train and self.use_sim:
            fpool_id, fpool_cloth = self.maskpooling(global_feat, mask)

        global_feat = pooling.MaxAvgPooling()(global_feat)
        global_feat = torch.flatten(global_feat, 1)
        id_feat = self.globalpooling(local_feat_chunk[0].transpose(1, 2))
        id_feat = torch.flatten(id_feat, 1)
        id_feat = id_feat + cls_tokens[0][:, 0, :]
        cloth_feat = self.globalpooling(local_feat_chunk[1].transpose(1, 2))
        cloth_feat = torch.flatten(cloth_feat, 1)
        cloth_feat = cloth_feat + cls_tokens[1][:, 0, :]

        f = self.bn(global_feat)
        f_id = self.bn_id(id_feat)
        f_cloth = self.bn_cloth(cloth_feat)
        
        if train and self.use_sim:
            fpool_id = self.bn_id(fpool_id)
            fpool_cloth = self.bn_cloth(fpool_cloth)
            return f, f_id, f_cloth, fpool_id, fpool_cloth

        return f, f_id, f_cloth


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