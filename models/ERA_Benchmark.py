import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from sklearn.linear_model import LinearRegression
from collections import Counter

class _LR(object):
    def __init__(self, 
                 fit_intercept,
                 normalize,
                 copy_X,
                 n_jobs,
                 flag
                ):
        super(_LR, self).__init__()
        
        if flag=='naive':
            self.init_model_naive = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)

        if flag=='iter':
            self.init_model_iter = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)
                    
        
class _FCN(nn.Module):
    def __init__(self, in_channel, hidden_channel, seq, flag): 
        super(_FCN, self).__init__()
        
        # Bottleneck
        if flag=='naive':
            self.encoder = nn.Sequential(
                BasicConv2d(in_channel, hidden_channel, 1, bn = True),
                BasicConv2d(hidden_channel, hidden_channel, 3, bn = True, padding = 1),
                BasicConv2d(hidden_channel, 128, 1, bn = True), 
            )
            self.shortcut = BasicConv2d(in_channel, 128, 1, bn=True)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                Flatten(),
                nn.Linear(128 * 1 * 1, 32),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(32, 1),
            )            

        if flag=='iter':
            self.seq = seq
            self.firstLayer = BasicConv2d(in_channel, hidden_channel, 1, bn = True)
            self.iterLayers = nn.ModuleList([BasicConv2d((in_channel+hidden_channel), hidden_channel, 3, bn = True, padding = 1)for seq_idx in range(self.seq-1)])
                                                                                                   
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                Flatten(),
                nn.Linear(hidden_channel * 1 * 1, 32),
                nn.ReLU(True),
                nn.Dropout(0.3),
                nn.Linear(32, 1),
            )               
            
    def forward(self, x, flag):                                     
        if flag=='naive':
            res = self.shortcut(x)
            x = self.encoder(x)
            x = x + res
            fcn_x = self.pool(x)
            fcn_x = self.fc(fcn_x)
            return fcn_x

        if flag=='iter':
            # first layer
            f_x = self.firstLayer(x[:,:,0,:,:])
            # iterative layer ↔ concat → N * C * H * W
            for seq_idx in range(self.seq-1):
                f_x = self.iterLayers[seq_idx](torch.cat((x[:,:,(seq_idx+1),:,:], f_x),1))
            #
            fcn_iter_x = self.pool(f_x)
            fcn_iter_x = self.fc(fcn_iter_x)
            return fcn_iter_x
  
                                                                                        
                                                   
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
                                                   
                                                   
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class Means(nn.Module):
    def __init__(self):
        super(Means, self).__init__()

    def forward(self, input):
        return torch.mean(input, dim=(1, 2, 3)).unsqueeze(-1)

        
class ZeroOuts(nn.Module):
    def __init__(self):
        super(ZeroOuts, self).__init__()

    def forward(self, x):
        batchSize = x.size()[0]
        return torch.zeros(batchSize, 4, 1, 1).cuda()