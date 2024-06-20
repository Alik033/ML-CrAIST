#!/usr/bin/env python3

import  math
import  torch
import  torch.nn as nn
import  torch.nn.functional as F
from    einops import rearrange
from    ops.OSAG import OSAG

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    #return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)
    return x_LL, x_HL, x_LH, x_HH

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  

    def forward(self, x):
        return dwt_init(x)

class LHFA(nn.Module):
    def __init__(self,num_in_ch=3, dim=64, num_heads=8, bias=False):
        super(LHFA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(num_in_ch, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.k = nn.Conv2d(num_in_ch, dim , kernel_size=1, bias=bias)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.v = nn.Conv2d(num_in_ch, dim , kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.fusion = nn.Conv2d(3*num_in_ch, dim , kernel_size=1, bias=bias)
        # self.fusion_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, x, y, z):
        b, c, h, w = x.shape
        concated = torch.cat((x,y,z), dim=1)
        q = self.q_dwconv(self.q(x))
        k = self.k_dwconv(self.k(y))
        v = self.v_dwconv(self.v(z))
        f = self.fusion(concated)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        
        return out + f
    
class CrossAttention(nn.Module):
    def __init__(self,num_in_ch=3, dim=64, num_heads=8, bias=False):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(num_in_ch, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(num_in_ch, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1) 

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out + x

class UpSampler(nn.Sequential):
    def __init__(self, scale, channel_num):

        m = []
        if scale == 8:
            kernel_size = 3
        elif scale == 16:
            kernel_size = 5
        else:
            kernel_size = 1

        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels=channel_num, out_channels=4 * channel_num, kernel_size=kernel_size, stride=1,
                                   padding=kernel_size // 2))
                m.append(nn.PixelShuffle(upscale_factor=2))
                m.append(nn.PReLU())
        super(UpSampler, self).__init__(*m)


class DWTAttention_L2(nn.Module):
    def __init__(self,num_in_ch=3, num_feat=64, bias = True, **kwargs):
        super(DWTAttention_L2,self).__init__()
        self.DWT = DWT()
        self.osag_in  = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.osag = OSAG(channel_num=num_feat, **kwargs)
        self.osag_out = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.fusion = LHFA(num_in_ch, num_feat)
        self.cross_attention = CrossAttention(num_feat, num_feat)
        self.up = UpSampler(2,num_feat)
        self.window_size   = kwargs["window_size"]
        

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        LL, HL, LH, HH = self.DWT(x)
        H, W = LL.shape[2:]
        LL, HL, LH, HH = self.check_image_size(LL), self.check_image_size(HL), self.check_image_size(LH), self.check_image_size(HH)

        osag_residual = self.osag_in(LL)
        osag_out = self.osag(osag_residual)
        osag_out = torch.add(self.osag_out(osag_out),osag_residual)

        hllhhh_ca = self.fusion(HL,LH,HH)
        low_high_ca = self.cross_attention(osag_out,hllhhh_ca)
        low_high_ca = low_high_ca[:, :, :H, :W]
        
        out = self.up(low_high_ca)

        return out


