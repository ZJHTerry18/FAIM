import torchvision
import torch
import copy
from torch import nn
from torch.nn import init
from models.utils import pooling
from models.utils.feature_block import MixupBlock, MixupBlock2, PatchShuffle
from models.utils.nonlocal_blocks import CADecouple
from models.vision_transformer_ import vit_l_16, vit_b_16
from models.vit_pytorch_ import vit_small_patch16_224_TransReID, vit_base_patch16_224_TransReID, vit_large_patch16_224_TransReID
from models.vit_pytorch_prompt import vit_base_patch16_224_PromptTransReID

vit_model_path = r'/home/zhaojiahe/.cache/torch/hub/checkpoints/jx_vit_base_p16_384-83fb41ba.pth'

class ViT_pytorch(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
        self.use_old_feature = config.MODEL.USE_OLD_FEATURE
        self.image_size = (config.DATA.HEIGHT, config.DATA.WIDTH)
        
        if config.MODEL.USE_PROMPT:
            vit = vit_base_patch16_224_PromptTransReID(img_size=self.image_size, stride_size=16, sie_xishu=0.0)
        else:
            vit = vit_base_patch16_224_TransReID(img_size=self.image_size, stride_size=16, sie_xishu=0.0)
        vit.load_param(model_path=vit_model_path)
        self.cls_token = vit.cls_token
        self.conv_proj = vit.patch_embed.proj
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm

        self.patchshuffle_block = PatchShuffle(patch_size=1, threshold=0.95)

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
    
    def forward(self, x, mask=None, train=False, idx_shuff=None):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        
        x = self.conv_proj(x)
        if train and self.do_patch_shuffle:
            # x_p = x.permute(0, 3, 1, 2).contiguous()
            x_p = self.patchshuffle_block(x, mask, idx_shuff=idx_shuff)
            # x_p = x_p.permute(0, 2, 3, 1).contiguous()
            if self.use_old_feature:
                x = torch.cat((x, x_p), dim=0)
            else:
                x = x_p

        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = x[:, 0]
        x_bn = self.bn(x)

        return x_bn


class ViTCA_pytorch(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
        self.use_old_feature = config.MODEL.USE_OLD_FEATURE
        self.image_size = (config.DATA.HEIGHT, config.DATA.WIDTH)

        vit = vit_base_patch16_224_TransReID(img_size=self.image_size, stride_size=16, sie_xishu=0.0, num_cls_token=2)
        vit.load_param(model_path=vit_model_path)
        self.cls_token = vit.cls_token
        self.conv_proj = vit.patch_embed.proj
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm

        self.patchshuffle_block = PatchShuffle(patch_size=1, threshold=0.95)

        self.bn_id = nn.BatchNorm1d(config.MODEL.CA_DIM)
        self.bn_cloth = nn.BatchNorm1d(config.MODEL.CA_DIM)
        init.normal_(self.bn_id.weight.data, 1.0, 0.02)
        init.constant_(self.bn_id.bias.data, 0.0)
        init.normal_(self.bn_cloth.weight.data, 1.0, 0.02)
        init.constant_(self.bn_cloth.bias.data, 0.0)
        
    
    def forward(self, x, mask=None, train=False, idx_shuff=None):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        
        x = self.conv_proj(x)
        if train and self.do_patch_shuffle:
            # x_p = x.permute(0, 3, 1, 2).contiguous()
            x_p = self.patchshuffle_block(x, mask, idx_shuff=idx_shuff)
            # x_p = x_p.permute(0, 2, 3, 1).contiguous()
            if self.use_old_feature:
                x = torch.cat((x, x_p), dim=0)
            else:
                x = x_p

        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x_id = x[:, 0]
        x_cloth = x[:, 1]
        x_id = self.bn_id(x_id)
        x_cloth = self.bn_cloth(x_cloth)
        x_all = torch.cat((x_id, x_cloth), dim=1)

        return x_all, x_id, x_cloth

class ViTCA_pytorch_2(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
        self.use_old_feature = config.MODEL.USE_OLD_FEATURE
        self.image_size = (config.DATA.HEIGHT, config.DATA.WIDTH)

        vit = vit_base_patch16_224_TransReID(img_size=self.image_size, stride_size=16, sie_xishu=0.0, num_cls_token=1)
        vit.load_param(model_path=vit_model_path)
        self.cls_token = vit.cls_token
        self.conv_proj = vit.patch_embed.proj
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.attlayer = nn.Sequential(
            copy.deepcopy(self.blocks[-1]),
            copy.deepcopy(self.norm)
        )

        self.patchshuffle_block = PatchShuffle(patch_size=1, threshold=0.95)

        self.bn_id = nn.BatchNorm1d(config.MODEL.CA_DIM)
        self.bn_cloth = nn.BatchNorm1d(config.MODEL.CA_DIM)
        init.normal_(self.bn_id.weight.data, 1.0, 0.02)
        init.constant_(self.bn_id.bias.data, 0.0)
        init.normal_(self.bn_cloth.weight.data, 1.0, 0.02)
        init.constant_(self.bn_cloth.bias.data, 0.0)
        
    
    def forward(self, x, mask=None, train=False, idx_shuff=None):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        
        x = self.conv_proj(x)
        if train and self.do_patch_shuffle:
            # x_p = x.permute(0, 3, 1, 2).contiguous()
            x_p = self.patchshuffle_block(x, mask, idx_shuff=idx_shuff)
            # x_p = x_p.permute(0, 2, 3, 1).contiguous()
            if self.use_old_feature:
                x = torch.cat((x, x_p), dim=0)
            else:
                x = x_p

        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        x_glb = x[:, 0:1]
        x_patch = x[:, 1:]
        patch_len = x_patch.size(1) // 2
        x_patch_id = x_patch[:, :patch_len]
        x_patch_cloth = x_patch[:, patch_len:]
        x_id = self.attlayer(torch.cat((x_glb, x_patch_id), dim=1))
        x_cloth = self.attlayer(torch.cat((x_glb, x_patch_cloth), dim=1))

        x_id = x_id[:, 0]
        x_cloth = x_cloth[:, 0]
        x_id = self.bn_id(x_id)
        x_cloth = self.bn_cloth(x_cloth)
        x_all = torch.cat((x_id, x_cloth), dim=1)

        return x_all, x_id, x_cloth

class ViTCA_pytorch_3(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
        self.use_old_feature = config.MODEL.USE_OLD_FEATURE
        self.image_size = (config.DATA.HEIGHT, config.DATA.WIDTH)

        vit = vit_base_patch16_224_TransReID(img_size=self.image_size, stride_size=16, sie_xishu=0.0, num_cls_token=2)
        vit.load_param(model_path=vit_model_path)
        self.cls_token = vit.cls_token
        self.conv_proj = vit.patch_embed.proj
        self.pos_embed_id = vit.pos_embed
        self.pos_embed_cloth = copy.deepcopy(self.pos_embed_id)
        self.pos_drop_id = vit.pos_drop
        self.pos_drop_cloth = copy.deepcopy(self.pos_drop_id)
        self.blocks_id = vit.blocks
        self.blocks_cloth = copy.deepcopy(self.blocks_id)
        self.norm_id = vit.norm
        self.norm_cloth = copy.deepcopy(self.norm_id)

        self.patchshuffle_block = PatchShuffle(patch_size=1, threshold=0.95)

        self.bn_id = nn.BatchNorm1d(config.MODEL.CA_DIM)
        self.bn_cloth = nn.BatchNorm1d(config.MODEL.CA_DIM)
        init.normal_(self.bn_id.weight.data, 1.0, 0.02)
        init.constant_(self.bn_id.bias.data, 0.0)
        init.normal_(self.bn_cloth.weight.data, 1.0, 0.02)
        init.constant_(self.bn_cloth.bias.data, 0.0)
        
    
    def forward(self, x, mask=None, train=False, idx_shuff=None):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        
        x = self.conv_proj(x)
        if train and self.do_patch_shuffle:
            # x_p = x.permute(0, 3, 1, 2).contiguous()
            x_p = self.patchshuffle_block(x, mask, idx_shuff=idx_shuff)
            # x_p = x_p.permute(0, 2, 3, 1).contiguous()
            if self.use_old_feature:
                x = torch.cat((x, x_p), dim=0)
            else:
                x = x_p

        x = x.flatten(2).transpose(1, 2)
        B = x.shape[0]
        cls_token_id = self.cls_token[0, 0, :].unsqueeze(0).expand(B, -1, -1)
        cls_token_cloth = self.cls_token[0, 1, :].unsqueeze(0).expand(B, -1, -1)
        x_id = torch.cat((cls_token_id, x), dim=1)
        x_cloth = torch.cat((cls_token_cloth, x), dim=1)

        x_id = x_id + self.pos_embed_id
        x_id = self.pos_drop_id(x_id)
        for blk in self.blocks_id:
            x_id = blk(x_id)
        x_id = self.norm_id(x_id)

        x_cloth = x_cloth + self.pos_embed_cloth
        x_cloth = self.pos_drop_cloth(x_cloth)
        for blk in self.blocks_cloth:
            x_cloth = blk(x_cloth)
        x_cloth = self.norm_cloth(x_cloth)

        x_id = x_id[:, 0]
        x_cloth = x_cloth[:, 0]
        x_id = self.bn_id(x_id)
        x_cloth = self.bn_cloth(x_cloth)
        x_all = torch.cat((x_id, x_cloth), dim=1)

        return x_all, x_id, x_cloth


class ViT(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.do_patch_shuffle = config.MODEL.PATCH_SHUFFLE
        self.use_old_feature = config.MODEL.USE_OLD_FEATURE
        self.image_size = (config.DATA.HEIGHT, config.DATA.WIDTH)

        vit = vit_b_16(weights='IMAGENET1K_V1', image_size=(config.DATA.HEIGHT, config.DATA.WIDTH))
        self.class_token = vit.class_token
        self.patch_size = vit.patch_size
        self.hidden_dim = vit.hidden_dim
        self.encoder = vit.encoder
        self.conv_proj = vit.conv_proj

        self.patchshuffle_block = PatchShuffle(patch_size=self.patch_size, threshold=0.95)

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
    
    def forward(self, x, mask=None, train=False, idx_shuff=None):
        # pre-processing step (include patch shuffle)
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size[0], "Wrong image height!")
        torch._assert(w == self.image_size[1], "Wrong image width!")
        n_h = h // p
        n_w = w // p
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        ## patch shuffle
        if train and self.do_patch_shuffle:
            x_p = x.permute(0, 3, 1, 2).contiguous()
            x_p = self.patchshuffle_block(x_p, mask, idx_shuff=idx_shuff)
            x_p = x_p.permute(0, 2, 3, 1).contiguous()
            if self.use_old_feature:
                x = torch.cat((x, x_p), dim=0)
            else:
                x = x_p
            print('x after patch shuffle:', x.shape)

        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)

        batch_size = x.shape[0]

        batch_class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        x = x[:, 0]
        # print('x output:', x.shape)
        x = self.bn(x)

        return x



