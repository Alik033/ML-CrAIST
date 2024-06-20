#!/usr/bin/env python3

import  torch
import  torch.nn as nn
from    ops.OSAG import OSAG
from    ops.DWTAttention import DWTAttention
from    ops.DWTAttention_L2 import CrossAttention
from    ops.pixelshuffle import pixelshuffle_block
import  torch.nn.functional as F
from    einops import rearrange


class ImageSR(nn.Module):
    def __init__(self,num_in_ch=3,num_out_ch=3,num_feat=64,**kwargs):
        super(ImageSR, self).__init__()

        res_num     = kwargs["res_num"]
        up_scale    = kwargs["upsampling"]
        bias        = kwargs["bias"]

        residual_layer  = []
        self.res_num    = res_num

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat,**kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input  = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up     = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)
        
        self.dwt_attention = DWTAttention(num_in_ch=num_in_ch,num_feat=num_feat,**kwargs)
        self.cross_attention = CrossAttention(num_feat, num_feat)

        # self.tail   = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, sqrt(2. / n))

        self.window_size   = kwargs["window_size"]
        self.up_scale = up_scale
        self.input_upscale = nn.Upsample(scale_factor=up_scale, mode='bicubic', align_corners=True)
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        input_upscale = self.input_upscale(x)
        H, W = x.shape[2:]
        x = self.check_image_size(x)


        residual= self.input(x)
        out     = self.residual_layer(residual)
        out     = torch.add(self.output(out),residual)
        
        fre     = self.dwt_attention(x)
        # out     = torch.add(fre,out)
        out = self.cross_attention(out,fre)
        
        out     = self.up(out)
        out = out[:, :, :H*self.up_scale, :W*self.up_scale]
        

        out = torch.add(out,input_upscale)
        return  out