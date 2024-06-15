import torchvision
import torch
from torch import nn
from torch.nn import init
import copy
import math
from models.utils import pooling
from models.utils.feature_block import PatchShuffle, FeatureClothErase, FeatureClothErase2, FeatureRenorm
from models.utils.nonlocal_blocks import CADecouple
from models.utils.pdecouple import LayerScale_Block_CA, LayerScale_Block_CA_2
from models.utils.maskpooling import ClothMaskPooling
from IPython import embed
      

class ResNet50(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
        self.erase_layer = config.MODEL.ERASE_LAYER
        self.use_old_feature = config.MODEL.USE_OLD_FEATURE

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
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        # self.fc = resnet50.fc

        # patch shuffle block
        self.patchshuffle_block = PatchShuffle(patch_size=8, threshold=0.95)

        # clothes-area erasing block
        in_channel_dict = {'layer1': 64, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        if config.MODEL.ERASE_LAYER in in_channel_dict.keys():
            self.clotherase_block = FeatureClothErase2(in_channels=in_channel_dict[config.MODEL.ERASE_LAYER],
                                                   p=config.AUG.RCE_PROB)

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
        
    def forward(self, x, mask=None, train=False, idx_shuff=None, output_hidden=False):
        # # only backbone
        # x = self.base(x)

        # backbone + mixup
        hidden_maps = dict()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        hidden_maps['conv'] = x
        if train and self.do_patch_shuffle:
            x_out = self.patchshuffle_block(x, mask, idx_shuff=idx_shuff)
            if self.use_old_feature:
                x = torch.cat((x, x_out), dim=0)
            else:
                x = x_out
        if train and self.erase_layer == 'layer1':
            x = self.clotherase_block(x, mask)
        x = self.layer1(x)
        hidden_maps['layer1'] = x
        x = self.layer2(x)
        hidden_maps['layer2'] = x
        if train and self.erase_layer == 'layer2':
            x = self.clotherase_block(x, mask)
        x = self.layer3(x)
        hidden_maps['layer3'] = x
        if train and self.erase_layer == 'layer3':
            x = self.clotherase_block(x, mask)
        x = self.layer4(x)
        hidden_maps['layer4'] = x
        if train and self.erase_layer == 'layer4':
            x = self.clotherase_block(x, mask)

        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        # calculate mean and variance for the whole hidden feature map and the clothes area feature map
        # fmap = hidden_maps['layer4']
        # b, c, h, w = fmap.size()
        # mask = ((mask==2) + (mask==3)).float().unsqueeze(1)
        # # resize mask to the same height and width as feature map
        # mask = (nn.functional.interpolate(mask, size=(h, w), mode='nearest') == 1).expand(b, c, h, w)

        # mean_i = fmap.flatten(2).mean(dim=2)
        # std_i = fmap.flatten(2).std(dim=2)
        # mean_c = torch.zeros(b, c).cuda()
        # std_c = torch.zeros(b, c).cuda()
        # for i in range(b):
        #     mean_c[i] = fmap[i].masked_select(mask[i]).view(c, -1).mean(dim=1) # [64,96,48] -> [64,]
        #     std_c[i] = fmap[i].masked_select(mask[i]).view(c, -1).std(dim=1)
        # # mean_c = fmap.masked_select(mask).view(b, -1).mean(dim=1)
        # # std_c = fmap.masked_select(mask).view(b, -1).std(dim=1)
        
        # stats = (mean_i, std_i, mean_c, std_c)
        return f

#1: Cross-attention version
class ResNet50CA(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
        self.use_old_feature = config.MODEL.USE_OLD_FEATURE

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
        # self.layer4_id_var = copy.deepcopy(self.layer4_main)
        self.layer4_clo_feat = copy.deepcopy(self.layer4_main)

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

        self.bn_main = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        self.bn_id = nn.BatchNorm1d(config.MODEL.CA_DIM)
        self.bn_cloth = nn.BatchNorm1d(config.MODEL.CA_DIM)
        init.normal_(self.bn_main.weight.data, 1.0, 0.02)
        init.constant_(self.bn_main.bias.data, 0.0)
        init.normal_(self.bn_id.weight.data, 1.0, 0.02)
        init.constant_(self.bn_id.bias.data, 0.0)
        init.normal_(self.bn_cloth.weight.data, 1.0, 0.02)
        init.constant_(self.bn_cloth.bias.data, 0.0)

    def forward(self, x, mask=None, train=False, idx_shuff=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # if train and self.do_patch_shuffle:
        #     x_out = self.patchshuffle_block(x, mask, idx_shuff=idx_shuff)
        #     if self.use_old_feature:
        #         x = torch.cat((x, x_out), dim=0)
        #     else:
        #         x = x_out

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
        # var_id = self.layer4_id_var(x)

        return f_main, f_id, f_cloth


# #2: Channel-divide version
# class ResNet50CA(nn.Module):
#     def __init__(self, config, **kwargs):
#         super().__init__()
#         self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
#         self.use_old_feature = config.MODEL.USE_OLD_FEATURE

#         resnet50 = torchvision.models.resnet50(pretrained=True)
#         if config.MODEL.RES4_STRIDE == 1:
#             resnet50.layer4[0].conv2.stride=(1, 1)
#             resnet50.layer4[0].downsample[0].stride=(1, 1) 
#         # self.base = nn.Sequential(*list(resnet50.children())[:-2])

#         # resnet50 detailed parts
#         self.conv1 = resnet50.conv1
#         self.bn1 = resnet50.bn1
#         self.relu = resnet50.relu
#         self.maxpool = resnet50.maxpool
#         self.layer1 = resnet50.layer1
#         self.layer2 = resnet50.layer2
#         self.layer3 = resnet50.layer3
#         self.layer4 = resnet50.layer4
#         # self.fc = resnet50.fc

#         # patch shuffle block
#         self.patchshuffle_block = PatchShuffle(patch_size=8, threshold=0.95)

#         if config.MODEL.POOLING.NAME == 'avg':
#             self.globalpooling = nn.AdaptiveAvgPool2d(1)
#         elif config.MODEL.POOLING.NAME == 'max':
#             self.globalpooling = nn.AdaptiveMaxPool2d(1)
#         elif config.MODEL.POOLING.NAME == 'gem':
#             self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
#         elif config.MODEL.POOLING.NAME == 'maxavg':
#             self.globalpooling = pooling.MaxAvgPooling()
#         else:
#             raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

#         self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
#         self.bn_id = nn.BatchNorm1d(config.MODEL.FEATURE_DIM // 2)
#         self.bn_cloth = nn.BatchNorm1d(config.MODEL.FEATURE_DIM // 2)
#         init.normal_(self.bn.weight.data, 1.0, 0.02)
#         init.constant_(self.bn.bias.data, 0.0)
#         init.normal_(self.bn_id.weight.data, 1.0, 0.02)
#         init.constant_(self.bn_id.bias.data, 0.0)
#         init.normal_(self.bn_cloth.weight.data, 1.0, 0.02)
#         init.constant_(self.bn_cloth.bias.data, 0.0)

#         # # cross-attention block
#         # self.ca_block = CADecouple(in_channels=config.MODEL.CA_DIM, out_channels=config.MODEL.CA_DIM,
#         #     num_heads=1, inter_channels=config.MODEL.CA_DIM // 2)


#     def forward(self, x, mask=None, train=False, idx_shuff=None):
#         # # only backbone
#         # x = self.base(x)

#         # backbone + mixup
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         if train and self.do_patch_shuffle:
#             x_out = self.patchshuffle_block(x, mask, idx_shuff=idx_shuff)
#             if self.use_old_feature:
#                 x = torch.cat((x, x_out), dim=0)
#             else:
#                 x = x_out

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         channel = x.size(1)
#         f_id = self.globalpooling(x[:,:channel // 2,:,:])
#         f_cloth = self.globalpooling(x[:,channel // 2:,:,:])
#         f = self.globalpooling(x)
#         f = f.view(f.size(0), -1)
#         f_id = f_id.view(f_id.size(0), -1)
#         f_cloth = f_cloth.view(f_cloth.size(0), -1)
#         f = self.bn(f)
#         f_id = self.bn_id(f_id)
#         f_cloth = self.bn_cloth(f_cloth)

#         # if not train:
#         #     return f, f_id, f_cloth, fmap

#         return f, f_id, f_cloth

#3: CA from UFDN, two branch, global by pooling
class ResNet50CA_2(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
        self.use_old_feature = config.MODEL.USE_OLD_FEATURE
        self.use_sim = config.MODEL.SIM
        self.num_decouple_blks = config.MODEL.NUM_DECOUPLE_BLOCKS
        self.feature_dim = config.MODEL.FEATURE_DIM
        self.ca_dim = config.MODEL.CA_DIM if config.MODEL.POOLING.NAME != 'maxavg' else config.MODEL.CA_DIM // 2
        self.pooling_type = config.MODEL.POOLING.NAME
        self.gap = True
        
        self.cls_token_list = []
        if not self.gap:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim // 2))
            self.cls_token_list.append(self.cls_token)
        for i in range(2):
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.ca_dim))
            self.cls_token_list.append(self.cls_token)
        # self.norm1 = nn.LayerNorm(config.MODEL.FEATURE_DIM)
        # self.norm2 = nn.LayerNorm(config.MODEL.FEATURE_DIM)
        # patch shuffle block
        self.patchshuffle_block = PatchShuffle(patch_size=8, threshold=0.95)
        if self.use_sim:
            self.maskpooling = ClothMaskPooling(ptype=config.MODEL.POOLING.NAME)

        # decoupling blocks
        if self.gap:
            self.decouple_blks = nn.ModuleList([
                LayerScale_Block_CA(
                    dim=config.MODEL.FEATURE_DIM // 2, num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                    num_stripes=2)
                for _ in range(self.num_decouple_blks)])
        else:
            self.decouple_blks = nn.ModuleList([
                LayerScale_Block_CA_2(
                    dim=config.MODEL.FEATURE_DIM // 2, num_heads=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                    num_stripes=2)
                for _ in range(self.num_decouple_blks)])
        
        self.apply(self._init_weights)

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
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.layer3_2 = copy.deepcopy(self.layer3)
        self.layer4_2 = copy.deepcopy(self.layer4)
        
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

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
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
        # # only backbone
        # x = self.base(x)

        # backbone + mixup
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
        x = self.layer2(x)

        B = x.size(0)
        global_feat = self.layer3(x)
        global_feat = self.layer4(global_feat)
        feat_dim = global_feat.size(1)

        local_feat = self.layer3_2(x)
        local_feat = self.layer4_2(local_feat)
        
        if not self.gap:
            global_feat = global_feat.view(B, feat_dim, -1).transpose(1, 2).contiguous()  # [b, h*w, c]
        local_feat = local_feat.view(B, feat_dim, -1).transpose(1, 2).contiguous()  # [b, h*w, c]

        # global_feat = self.norm1(global_feat)
        # local_feat = self.norm2(local_feat)
        local_feat_chunk = local_feat.chunk(2, dim=2)   # [b, h*w, c/2]

        # decoupling attention
        cls_tokens = []
        for i in range(len(self.cls_token_list)):
            cls_token_tmp = self.cls_token_list[i].expand(B, -1, -1).cuda()
            cls_tokens.append(cls_token_tmp)
        
        for i, blk in enumerate(self.decouple_blks):
            cls_tokens = blk(global_feat, local_feat_chunk, cls_tokens)
        
        if train and self.use_sim:
            fpool_id, fpool_cloth = self.maskpooling(global_feat, mask)

        if self.gap:
            global_feat = pooling.MaxAvgPooling()(global_feat)
        else:
            global_feat = self.globalpooling(global_feat.transpose(1, 2))
        global_feat = torch.flatten(global_feat, 1)
        id_feat = self.globalpooling(local_feat_chunk[0].transpose(1, 2))
        id_feat = torch.flatten(id_feat, 1)
        cloth_feat = self.globalpooling(local_feat_chunk[1].transpose(1, 2))
        cloth_feat = torch.flatten(cloth_feat, 1)
        if not self.gap:
            if self.pooling_type == 'maxavg':
                global_feat = global_feat + torch.cat((cls_tokens[0][:, 0, :], cls_tokens[0][:, 0, :]), dim=1)
                id_feat = id_feat + torch.cat((cls_tokens[1][:, 0, :], cls_tokens[1][:, 0, :]), dim=1)
                cloth_feat = cloth_feat + torch.cat((cls_tokens[2][:, 0, :], cls_tokens[2][:, 0, :]), dim=1)
            else:
                global_feat = global_feat + cls_tokens[0][:, 0, :]
                id_feat = id_feat + cls_tokens[1][:, 0, :]
                cloth_feat = cloth_feat + cls_tokens[2][:, 0, :]
        else:
            if self.pooling_type == 'maxavg':
                id_feat = id_feat + torch.cat((cls_tokens[0][:, 0, :], cls_tokens[0][:, 0, :]), dim=1)
                cloth_feat = cloth_feat + torch.cat((cls_tokens[1][:, 0, :], cls_tokens[1][:, 0, :]), dim=1)
            else:
                id_feat = id_feat + cls_tokens[0][:, 0, :]
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