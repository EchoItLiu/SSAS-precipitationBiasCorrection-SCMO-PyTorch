from __future__ import absolute_import, division
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
sys.path.append("/home/yqliu/EC-BiasCorrecting/pre_training")

'''
MTM
'''
# 把通道数变为MSM输出的通道大小；且用adaptivePooling把尺度统一
class middleFeaturesCNN(nn.Module):
    def __init__(self, in_channel, uni_scale, ED_out_channel):
        super(middleFeaturesCNN, self).__init__()
        self.conv1 = BasicConv2d(in_channel, 256, 1, bn=True)
        self.conv2 = BasicConv2d(256, 256, 3, bn=True, padding=1)        
        self.conv3 = BasicConv2d(256, ED_out_channel, 1, bn=True)
        self.uni_scale = uni_scale
        self.ED_out_channel = ED_out_channel
        self.uniDown = nn.AdaptiveMaxPool2d(uni_scale)
        
    def forward(self, x, isTrain):
        # N1 * C * D * H * W → N1 * D * C * H * W → N1D * C * H * W
        Bs = x.size(0)
        feature_cha = x.size(1)
        feature_len = x.size(2)
        hw = x.size(3)
        ND_x = x.transpose(1,2).contiguous().view(-1, feature_cha, hw, hw)
        #
        statdf_cuda = self.conv1(ND_x)
        statdf_cuda = self.conv2(statdf_cuda)
        statdf_cuda = self.conv3(statdf_cuda)        
        # uniform scale
        statdf_cuda = self.uniDown(statdf_cuda)           
        # N1D * C1 * H * W → N1 * D * C1 * H * W →  N1 * C1 * D * H * W  
        gpu_midF_stats = statdf_cuda.contiguous().view(Bs, feature_len, self.ED_out_channel, self.uni_scale, self.uni_scale).transpose(1,2)
        return gpu_midF_stats 

# class C3AE_3DI   
class C3AE_3DI(nn.Module):
    def __init__(self, seq, Range, curscale, ED_out_channel):
        super(C3AE_3DI, self).__init__()
        self.curscale = curscale
        prob_len = len(Range)
        # kernal depth 2 and shared weight
        self.Conv_3d = BasicConv3d(ED_out_channel, 64, (2,3,3), bn=True, padding=(0,1,1))
        # Pooling
#         self.pooling  = nn.AdaptiveAvgPool2d(output_size=(1, 1))     
        # crop
        self.fc_prob = nn.Linear(64*curscale*curscale, prob_len)
        self.fc_pred = nn.Linear(prob_len, 1)
                                  
    def forward(self, x):
        # N1*C1*D*H_u*W_u: D → D-1 ... → 1
        flag = 0
        while (x.size(2)) >1:
            x = self.Conv_3d(x)
#             print ('curscale:', x.size(2))
        x = torch.flatten(x, start_dim=1)        
        # N1 * len(range)                         
        prob = self.fc_prob(x)
        # N1                         
        pred = self.fc_pred(prob)                          
        return prob, pred
                         
# class C3AE_3DII
'''
之前MSM的wind的写法有问题，应该单独两个分支FC然后都输出1进行loss，而不是输出2，记得改
'''                                  
class C3AE_3DII(nn.Module):
    def __init__(self, seq, wdRange, wsRange, curscale, ED_out_channel):
        super(C3AE_3DII, self).__init__()
        self.curscale = curscale
        prob_len_wd = len(wdRange)
        prob_len_ws = len(wsRange)
        
        self.curscale = curscale
        
        # kernal depth 2 and shared weight
        self.Conv_3d = BasicConv3d(ED_out_channel, 64, (2,3,3), bn=True, padding=(0,1,1))
        
        # Pooling
#         self.pooling  = nn.AdaptiveAvgPool2d(output_size=(1, 1))     
        # crop
        self.fc_wd_prob = nn.Linear(64*curscale*curscale, prob_len_wd)
        self.fc_ws_prob = nn.Linear(64*curscale*curscale, prob_len_ws)                   
        self.fc_wd_pred = nn.Linear(prob_len_wd, 1)
        self.fc_ws_pred = nn.Linear(prob_len_ws, 1)
                                                                    
    def forward(self, x):
        # N1*C1*D*H_u*W_u: D → D-1 ... → 1
        flag = 0
        while (x.size(2)) >1:
            x = self.Conv_3d(x)
#             print ('curscale:', x.size(2))
        x = torch.flatten(x, start_dim=1)        
        # N1 * len(range) 
        # N1*C(HW_u)                         
        prob_wd = self.fc_wd_prob(x)
        prob_ws = self.fc_ws_prob(x)
        # N1                         
        pred_wd = self.fc_wd_pred(prob_wd)
        pred_ws = self.fc_ws_pred(prob_ws)              
        return prob_wd, pred_wd, prob_ws, pred_ws

    
## util classes
class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 relu = True, bn = False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        
        if bn:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            # 
            x = F.leaky_relu(x, inplace = True)
        return x  


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 relu = True, bn = False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            # 
            x = F.leaky_relu(x, inplace = True)
        return x
                                  
class cropConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 relu = True, bn = False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.relu = relu
        
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            # 
            x = F.leaky_relu(x, inplace = True)
        return x  
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, _input):
        return _input.view(_input.size(0), -1)