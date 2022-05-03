# import torch.nn as nn
# import torch
# from torch.nn import init
# from .submodules import *

# class FlowSD(nn.Module):
#     def __init__(self, batchNorm=True):
#         super(FlowSD,self).__init__()

#         self.batchNorm = batchNorm
#         self.conv0   = conv(self.batchNorm,  6,   64)
#         self.conv1   = conv(self.batchNorm,  64,   64, stride=2)
#         self.conv1_1 = conv(self.batchNorm,  64,   128)
#         self.conv2   = conv(self.batchNorm,  128,  128, stride=2)
#         self.conv2_1 = conv(self.batchNorm,  128,  256)
#         self.conv3   = conv(self.batchNorm, 256,  256, stride=2)
#         self.conv3_1 = conv(self.batchNorm, 256,  512)
#         self.conv4   = conv(self.batchNorm, 512,  512, stride=2)
#         self.conv4_1 = conv(self.batchNorm, 512,  1024)
#         self.conv_mid_1 = conv(self.batchNorm, 1024, 1024)
#         self.conv_mid_2 = conv(self.batchNorm, 1024, 1024)
#         self.conv_mid_3 = conv(self.batchNorm, 1024, 1024)
#         self.conv_mid_4 = conv(self.batchNorm, 1024, 512)

#         self.deconv3 = deconv(512,256)
#         self.deconv2 = deconv(256,128)
#         self.deconv1 = deconv(128,64)
#         self.deconv0 = deconv(64,32)

#         self.inter_conv4 = i_conv(self.batchNorm,  1536,  512)
#         self.inter_conv3 = i_conv(self.batchNorm,  768,  256)
#         self.inter_conv2 = i_conv(self.batchNorm,  384,  128)
#         self.inter_conv1 = i_conv(self.batchNorm,  192,  64)

#         self.out = predict_flow(32)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.bias is not None:
#                     init.uniform_(m.bias)
#                 init.xavier_uniform_(m.weight)

#             if isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     init.uniform_(m.bias)
#                 init.xavier_uniform_(m.weight)
#                 # init_deconv_bilinear(m.weight)


#     def forward(self, x):
#         out_conv0 = self.conv0(x)
#         out_conv1 = self.conv1_1(self.conv1(out_conv0))
#         out_conv2 = self.conv2_1(self.conv2(out_conv1))

#         out_conv3 = self.conv3_1(self.conv3(out_conv2))
#         out_conv4 = self.conv4_1(self.conv4(out_conv3))

#         out_mid = self.conv_mid_1(out_conv4)
#         out_mid = self.conv_mid_2(out_mid)
#         out_mid = self.conv_mid_3(out_mid)
#         out_mid = self.conv_mid_4(out_mid)

#         concat4 = torch.cat((out_conv4,out_mid),1)
#         out_interconv4 = self.inter_conv4(concat4)
#         out_deconv3 = self.deconv3(out_interconv4)

#         concat3 = torch.cat((out_conv3,out_deconv3),1)
#         out_interconv3 = self.inter_conv3(concat3)
#         out_deconv2 = self.deconv2(out_interconv3)

#         concat2 = torch.cat((out_conv2,out_deconv2),1)
#         out_interconv2 = self.inter_conv2(concat2)
#         out_deconv1 = self.deconv1(out_interconv2)

#         concat1 = torch.cat((out_conv1,out_deconv1),1)
#         out_interconv1 = self.inter_conv1(concat1)
#         out_deconv0 = self.deconv0(out_interconv1)

#         out = self.out(out_deconv0)

#         return out




#########################################################################################

import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from .submodules import *
'Parameter count = 45,371,666'

class FlowSD(nn.Module):
    def __init__(self, args, batchNorm=True):
        super(FlowSD,self).__init__()

        self.batchNorm = batchNorm
        self.conv0   = conv(self.batchNorm,  6,   64)
        self.conv1   = conv(self.batchNorm,  64,   64, stride=2)
        self.conv1_1 = conv(self.batchNorm,  64,   128)
        self.conv2   = conv(self.batchNorm,  128,  128, stride=2)
        self.conv2_1 = conv(self.batchNorm,  128,  128)
        self.conv3   = conv(self.batchNorm, 128,  256, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.inter_conv5 = i_conv(self.batchNorm,  1026,   512)
        self.inter_conv4 = i_conv(self.batchNorm,  770,   256)
        self.inter_conv3 = i_conv(self.batchNorm,  386,   128)
        self.inter_conv2 = i_conv(self.batchNorm,  194,   64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')



    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5       = self.predict_flow5(out_interconv5)

        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4       = self.predict_flow4(out_interconv4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3       = self.predict_flow3(out_interconv3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2
#########################################################################################


def get_grid(batchsize, rows, cols, gpu_id=0, dtype=torch.float32):
    hor = torch.linspace(-1.0, 1.0, cols)
    hor.requires_grad = False
    hor = hor.view(1, 1, 1, cols)
    hor = hor.expand(batchsize, 1, rows, cols)
    ver = torch.linspace(-1.0, 1.0, rows)
    ver.requires_grad = False
    ver = ver.view(1, 1, rows, 1)
    ver = ver.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([hor, ver], 1)
    t_grid.requires_grad = False

    if dtype == torch.float16: t_grid = t_grid.half()
    return t_grid.cuda(gpu_id)

def resample(image, flow, grid):
    b, c, h, w = image.size()
    if grid is None or grid.size() != flow.size():
        grid = get_grid(b, h, w, gpu_id=flow.get_device(), dtype=flow.dtype)
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
    final_grid = (grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
    output = torch.nn.functional.grid_sample(image, final_grid, mode='bilinear', padding_mode='border')
    return output, grid

