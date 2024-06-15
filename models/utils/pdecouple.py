import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc as container_abcs
import torch.utils.checkpoint as checkpoint

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    
    def forward(self, x ):
        
        B, N, C = x.shape
        # print(x.size(),B, 1, self.num_heads, C // self.num_heads)
        # exit()
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls

class LayerScale_Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
                 Mlp_block=Mlp,init_values=1e-4,num_stripes=1):
        super().__init__()
        # self.attn = Attention_block(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn2 = Attention_block(
            int(dim/num_stripes), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.num_stripes=num_stripes
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp_block(in_features=int(dim/num_stripes), hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_list = nn.ModuleList()
        # self.gamma_global = nn.Parameter(init_values * torch.ones((2, dim)))
        self.gamma_local = nn.Parameter(init_values * torch.ones((num_stripes * 2, dim // num_stripes)))
        # for i in range(2):
        #     self.norm_list.append(norm_layer(dim))
        for i in range(num_stripes*2):
            self.norm_list.append(norm_layer(int(dim/num_stripes)))
    
    def forward(self, global_feat,local_feats, cls_tokens):
        # u1 = torch.cat((cls_tokens[0],global_feat),dim=1)
        u_feats=[]
        for i in range(len(local_feats)):
            u_tmp=torch.cat((cls_tokens[i],local_feats[i]),dim=1)
            u_feats.append(u_tmp)  
        # x_cls1 = cls_tokens[0] + self.drop_path(self.gamma_global[0] * self.attn(u1))
        # x_cls1 = x_cls1 + self.drop_path(self.gamma_global[1] * self.mlp(self.norm_list[0](x_cls1)))
        # x_cls_list=[self.norm_list[1](x_cls1)]
        x_cls_list = []
        for i in range(self.num_stripes):
            x_cls2 = cls_tokens[i] + self.drop_path(self.gamma_local[i*2] * self.attn2(u_feats[i]))
            x_cls2 = x_cls2 + self.drop_path(self.gamma_local[1+i*2] * self.mlp2(self.norm_list[i*2](x_cls2)))
            x_cls_list.append(self.norm_list[1+i*2](x_cls2))
        return x_cls_list

class LayerScale_Block_CA_2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
                 Mlp_block=Mlp,init_values=1e-4,num_stripes=1):
        super().__init__()
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn2 = Attention_block(
            int(dim/num_stripes), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.num_stripes=num_stripes
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp_block(in_features=int(dim/num_stripes), hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_list = nn.ModuleList()
        self.gamma_global = nn.Parameter(init_values * torch.ones((2, dim)))
        self.gamma_local = nn.Parameter(init_values * torch.ones((num_stripes * 2, dim // num_stripes)))
        for i in range(2):
            self.norm_list.append(norm_layer(dim))
        for i in range(num_stripes*2):
            self.norm_list.append(norm_layer(int(dim/num_stripes)))
    
    def forward(self, global_feat,local_feats, cls_tokens):
        u1 = torch.cat((cls_tokens[0],global_feat),dim=1)
        u_feats=[u1]
        for i in range(len(local_feats)):
            u_tmp=torch.cat((cls_tokens[i+1],local_feats[i]),dim=1)
            u_feats.append(u_tmp)  
        x_cls1 = cls_tokens[0] + self.drop_path(self.gamma_global[0] * self.attn(u1))
        x_cls1 = x_cls1 + self.drop_path(self.gamma_global[1] * self.mlp(self.norm_list[0](x_cls1)))
        x_cls_list=[self.norm_list[1](x_cls1)]
        for i in range(self.num_stripes):
            x_cls2 = cls_tokens[i+1] + self.drop_path(self.gamma_local[i*2] * self.attn2(u_feats[i+1]))
            x_cls2 = x_cls2 + self.drop_path(self.gamma_local[1+i*2] * self.mlp2(self.norm_list[2+i*2](x_cls2)))
            x_cls_list.append(self.norm_list[3+i*2](x_cls2))
        return x_cls_list