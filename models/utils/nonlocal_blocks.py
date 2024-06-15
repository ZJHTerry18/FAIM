import torch
import math
from torch import nn
from torch.nn import functional as F
from models.utils import inflate

class CABlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, inter_channels=None, bn_layer=True):
        super(CABlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.inter_channels = inter_channels
        self.num_dec = 2 # number of decoupled features

        if self.inter_channels is None:
            self.inter_channels = max(in_channels // 2, 1)
        # assert self.inter_channels % self.num_heads == 0
        self.scale = self.inter_channels ** -0.5

        # define conv blocks
        self.w_q_list = nn.ModuleList([
            nn.Conv1d(self.out_channels, self.inter_channels * self.num_heads, kernel_size=1, stride=1, padding=0, bias=True, groups=self.num_heads) for _ in range(self.num_dec)
        ]) # one projection matrix for each token
        # self.w_q = nn.Conv1d(self.out_channels, self.inter_channels * self.num_heads, kernel_size=1, stride=1, padding=0, bias=True)
        self.w_k = nn.Conv2d(self.in_channels, self.inter_channels * self.num_heads, kernel_size=1, stride=1, padding=0, bias=True, groups=self.num_heads)
        self.w_v = nn.Conv2d(self.in_channels, self.inter_channels * self.num_heads, kernel_size=1, stride=1, padding=0, bias=True, groups=self.num_heads)
        
        if bn_layer:
            self.w_f = nn.Sequential(
                nn.Conv1d(self.inter_channels * self.num_heads, self.out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(self.out_channels)
            )
        else:
            self.w_f = nn.Conv1d(self.inter_channels * self.num_heads, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(self.out_channels)

        # init
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        # if bn_layer:
        #     nn.init.constant_(self.w_f[1].weight.data, 0.0)
        #     nn.init.constant_(self.w_f[1].bias.data, 0.0)
        # else:
        #     nn.init.constant_(self.w_f.weight.data, 0.0)
        #     nn.init.constant_(self.w_f.bias.data, 0.0)
        
        # nn.init.normal_(self.bn.weight.data, 1.0, 0.02)
        # nn.init.constant_(self.bn.bias.data, 0.0)
        
    def forward(self, x, tokens):
        '''
            inputs:
                x: feature map, (B,C_in,H,W)
                tokens: feature tokens before cross-attention, (B,2,C_out)
            outputs:
                f: feature tokens after cross-attention, (B,2,C_out)
        '''
        batch_size = x.size(0)
        if len(tokens.shape) < 3:
            tokens = tokens.repeat(batch_size, 1, 1)

        q = torch.zeros((batch_size, self.num_dec, self.inter_channels * self.num_heads)).cuda()
        for i in range(self.num_dec):
            q[:,i,:] = self.w_q_list[i](tokens[:,i,:].unsqueeze(-1)).squeeze()
        # q = self.w_q(tokens.transpose(1,2)).transpose(1,2)
        q = q.view(batch_size, self.num_dec, self.num_heads, -1).permute(0,2,1,3)                     # q:(B, n_h, 2, d])
        k = self.w_k(x).view(batch_size, self.num_heads, self.inter_channels, -1)                     # k:(B, n_h, d, H*W)
        v = self.w_v(x).view(batch_size, self.num_heads, self.inter_channels, -1).transpose(-2,-1)    # v:(B, n_h, h*W, d)
        # print('q:', q.shape, 'k:', k.shape, 'v:', v.shape)

        # calculate cross-attention
        att = torch.matmul(q, k) * self.scale
        att = F.softmax(att, dim=-1)
        f = torch.matmul(att, v)    # f:(B, n_h, 2, d)
        f = f.transpose(-2,-1).contiguous().view(batch_size, self.inter_channels * self.num_heads, self.num_dec)    # f:(B, d*n_h, 2)
        f = self.w_f(f)             # f:(B, C_out, 2)
        f = f + tokens.transpose(1,2)
        f = self.bn(f).transpose(1,2)
        # print('f:', f.shape)

        return f


class CADecouple(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, num_heads=1, inter_channels=None):
        super(CADecouple, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.inter_channels = inter_channels
        self.num_dec = 2 # number of decoupled features
                
        # define feature tokens, for cloth-irrelevant & cloth-relevant features
        self.tokens = nn.Parameter(torch.zeros(self.num_dec, self.out_channels))   # tokens:(C_out, num_dec)

        self.ca_blocks = nn.ModuleList([
            CABlock(in_channels=self.in_channels, out_channels=self.in_channels, num_heads=self.num_heads, 
            inter_channels=self.inter_channels) for _ in range(num_blocks)
            ])   

        
    def forward(self, x):
        '''
            inputs:
                x: feature map, (B,C_in,H,W)
            outputs:
                f_id: id-relevant feature, (B,C_out)
                f_cloth: cloth-relevant feature, (B,C_out)
        '''
        batch_size = x.size(0)

        tokens = self.tokens
        for i in range(self.num_blocks):
            tokens = self.ca_blocks[i](x, tokens)
            # print('token %d' % (i), tokens.shape)

        tokens = tokens.transpose(0,1)  # tokens:(2, B, C_out)
        f_id = tokens[0,:,:]
        f_cloth = tokens[1,:,:]
        # print('f_id:', f_id.shape, 'f_cloth:', f_cloth.shape)

        return f_id, f_cloth


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            # max_pool = inflate.MaxPool2dFor3dInput
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(self.in_channels, self.inter_channels,
                         kernel_size=1, stride=1, padding=0, bias=True)
        self.theta = conv_nd(self.in_channels, self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.phi = conv_nd(self.in_channels, self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        # if sub_sample:
        #     self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
        #     self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))
        if sub_sample:
            if dimension == 3:
                self.g = nn.Sequential(self.g, max_pool((1, 2, 2)))
                self.phi = nn.Sequential(self.phi, max_pool((1, 2, 2)))
            else:
                self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(self.inter_channels, self.in_channels,
                        kernel_size=1, stride=1, padding=0, bias=True),
                bn(self.in_channels)
            )
        else:
            self.W = conv_nd(self.inter_channels, self.in_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        
        # init
        for m in self.modules():
            if isinstance(m, conv_nd):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if bn_layer:
            nn.init.constant_(self.W[1].weight.data, 0.0)
            nn.init.constant_(self.W[1].bias.data, 0.0)
        else:
            nn.init.constant_(self.W.weight.data, 0.0)
            nn.init.constant_(self.W.bias.data, 0.0)


    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f = F.softmax(f, dim=-1)

        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        y = self.W(y)
        z = y + x

        return z


class NonLocalBlock1D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NonLocalBlock2D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NonLocalBlock3D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
