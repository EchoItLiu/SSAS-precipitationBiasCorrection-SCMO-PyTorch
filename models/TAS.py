import torch
import torch.nn as nn
from pre_training.deformConv.layers import ConvOffset2D
from apex.parallel import SyncBatchNorm
import numpy as np
from torch.nn import functional as F
 
class Encoder(nn.Module):
    def __init__(self, uniScale, in_channel, ed_out_channel):
        super(Encoder, self).__init__()
        self.uniScale = uniScale
        # 考虑one sample BN是否得False
        self.shortcut = BasicConv2d(in_channel, 256, 1, bn=True, padding=0)
        # Bottleneck
        self.bk = nn.Sequential(
            BasicConv2d(256, 64, 1, bn=True),
            BasicConv2d(64, 64, 3, bn=True, padding=1),
            BasicConv2d(64, 256, 1, bn=True),
            )
        # smooth
        self.smo = BasicConv2d(256, ed_out_channel, 3, bn=True, padding=1)
        # pooling
        self.uniDown = nn.AdaptiveMaxPool2d(uniScale)
        
    def forward(self, x, Bs, isTrain):
        cpu_stats_l = []
        # N * C * D * H * W → N * D * C * H * W → ND * C * H * W
        feature_cha = x.size(1)
        hw = x.size(3)
        ND_x = x.transpose(1,2).contiguous().view(-1, feature_cha, hw, hw)        
        
        for stat_cuda in ND_x:
            stat_cuda = stat_cuda.unsqueeze(0)
            stat_cuda = self.shortcut(stat_cuda)
            feature_stat = self.bk(stat_cuda) + stat_cuda
            feature_stat = self.smo(feature_stat)
            feature_stat = self.uniDown(feature_stat)
            if isTrain:
                feature_stat = feature_stat.detach()

            cpu_stats_l.append(feature_stat.cpu().numpy())
        # N*D*1*C*H*W → N*D*C*H*W → N*D*C*H*W → N*C*D*H*W  
        cpu_stats = np.stack(cpu_stats_l).squeeze(1)
        channel_size = cpu_stats.shape[1]
        # 
        cpu_stats = np.reshape(cpu_stats, (Bs, -1, channel_size, self.uniScale, self.uniScale))
        cpu_stats = np.transpose(cpu_stats, (0,2,1,3,4))
        return cpu_stats
        
# class forward CNN_3D
class CNN_3D(nn.Module):
    def __init__(self, curscale, ed_out_channel, out_channel_3d):
        super(CNN_3D, self).__init__()
        self.curscale = curscale
        self.Conv_3d = BasicConv3d(ed_out_channel, out_channel_3d, (2,3,3), bn=True, padding=(0,1,1))
                                  
    def forward(self, x, isTrain):
        cpu_out_stats_l = []
        for stat_ts in x:
#             stat_ts = stat_ts.unsqueeze(0)
            # 1*C1*D*H_u*W_u: D → D-1 ... → 1            
            while (stat_ts.size(2)) >1:
                stat_ts = self.Conv_3d(stat_ts)                
            if isTrain:
                stat_ts = stat_ts.detach()
            
            cpu_out_stats_l.append(stat_ts.cpu().numpy())
        # N*1*C*H*W → N*C*H*W
        cpu_out_stats = np.stack(cpu_out_stats_l).squeeze(1).squeeze(2)
        out_stats = torch.from_numpy(cpu_out_stats.astype(np.float32)).cuda()
        return out_stats 
        
class OrdinalRegressionModel(nn.Module):
    def __init__(self, nClass):
        super(OrdinalRegressionModel, self).__init__()
        self.nClass = nClass
        self.boosting = nn.ModuleList()

        for i in range(self.nClass):
            oneClassifier = BaseClassifier()
            self.boosting.append(oneClassifier)

    def forward(self, x): 
#         print ('self.nClass:', self.nClass)
        # list sigmoid outputs from all classifers
        outputs = [self.boosting[i](x) for i in range(self.nClass)]
        # list → Tensor (torch.Size([1, nClass])
        return torch.cat(outputs, dim = 1)
        

class rainFallClassification(nn.Module):
    def __init__(self):
        super(rainFallClassification, self).__init__()
        self.conv = nn.Sequential(
            BasicConv2d(57, 64, 3, bn = True, padding=1),
            BasicConv2d(64, 64, 3, bn = True, padding=1),
            BasicConv2d(64, 128, 3, bn = True, padding=1),
            BasicConv2d(128, 128, 3, bn = True, padding=1),
        )

        self.downsample = nn.Sequential(
            BasicConv2d(57, 128, 1, bn = True, relu = False, padding=0)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(128 * 1 * 1, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv(x)
        x = x + residual
        x = self.pool(x)
        x = self.fc(x)
        return x

    
   
    
## util classes
class BaseClassifier(nn.Module):
    def __init__(self):
        super(BaseClassifier, self).__init__()
        self.conv = nn.Sequential(
            BasicConv2d(64, 128, 3, padding = 1),
            BasicConv2d(128, 128, 3, padding = 1),
            BasicConv2d(128, 32, 1),
            BasicConv2d(32, 32, 1),
        )

        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.downsampler = nn.Sequential(
            nn.Conv2d(64, 32, 1, padding=0),
            SyncBatchNorm(32).cuda()
#             nn.BatchNorm2d(32)
        )

        self.ac = nn.ReLU(True)
        # sigmoid for multi-labels and outputing a probability
        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )  
        
    def forward(self, x):
        identity = self.downsampler(x)
        x = self.conv(x)
        x = self.ac(x + identity)
        x = self.pooling(x)
        x = self.fc(x)
        return x        
        
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 relu = True, bn = False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        
        if bn:
            self.bn = SyncBatchNorm(out_channels).cuda()
#             self.bn = nn.BatchNorm2d(out_channels)
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
    




