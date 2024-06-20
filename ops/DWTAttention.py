#!/usr/bin/env python3

import  math
import  torch
import  torch.nn as nn
import  torch.nn.functional as F
from    einops import rearrange
from    ops.OSAG import OSAG
from    ops.DWTAttention_L2 import DWTAttention_L2, DWT, LHFA, CrossAttention, UpSampler

class DWTAttention(nn.Module):
    def __init__(self,num_in_ch=3, num_feat=64, num_level=1, bias = True, **kwargs):
        super(DWTAttention,self).__init__()
        self.num_in_ch= num_in_ch
        self.num_feat = num_feat
        # self.args= **kwargs 
        self.DWT = DWT()
        self.osag_in  = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.osag = OSAG(channel_num=num_feat, **kwargs)
        self.osag_out = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.fusion = LHFA(num_in_ch, num_feat)
        self.cross_attention1 = CrossAttention(num_feat, num_feat)
        self.cross_attention2 = CrossAttention(num_feat, num_feat)
        self.up = UpSampler(2,num_feat)
        self.window_size   = kwargs["window_size"]
        self.dwt_attention = DWTAttention_L2(num_in_ch=num_in_ch,num_feat=num_feat,**kwargs)
        self.dwt_level = num_level
        # self.up_scale = kwargs["upsampling"]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x


    def forward(self, x):
        if(self.dwt_level==1):
            LL, HL, LH, HH = self.DWT(x)
            LL_LL = self.dwt_attention(LL)
            
            H, W = LL.shape[2:]
            LL, HL, LH, HH = self.check_image_size(LL), self.check_image_size(HL), self.check_image_size(LH), self.check_image_size(HH)

            osag_residual = self.osag_in(LL)
            osag_out = self.osag(osag_residual)
            osag_out = torch.add(self.osag_out(osag_out),osag_residual)

            hllhhh_ca = self.fusion(HL,LH,HH)
            low_high_ca = self.cross_attention1(osag_out,hllhhh_ca)
            low_high_ca = low_high_ca[:, :, :H, :W]
            
            # merged = torch.add(low_high_ca, LL_LL)
            merged = self.cross_attention2(low_high_ca, LL_LL)

            # out = self.up(low_high_ca)

            out = self.up(merged)
            return out
        
        


