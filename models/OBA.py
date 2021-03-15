import torch
import torch.nn as nn
from apex.parallel import SyncBatchNorm
import numpy as np
from torch.nn import functional as F

"""------- flat according to channel -------"""
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


class NormalizerLoss(nn.Module):
    def __init__(self, std):
        super(NormalizerLoss, self).__init__()
        self.std = torch.from_numpy(
            np.array(std).astype('float32')
        ).cuda()
        self.lambda_ = 1e-2

    def forward(self, x, y, rainMask, regressionMask):
        se = ((x - y) / self.std) ** 2
        se = torch.matmul(regressionMask.unsqueeze(-2), se.unsqueeze(-1)).squeeze(-1)
        zeros = torch.zeros(se.size()).to(se.device)
        se = torch.cat([zeros, se], dim=1).unsqueeze(-1)
        rainMask = rainMask.unsqueeze(-2)
        loss = torch.matmul(rainMask, se)

        return loss.mean()


class MeanVarLoss(nn.Module):
    def __init__(self, nClass):
        super(MeanVarLoss, self).__init__()
        self.labels = torch.arange(nClass).view(nClass, 1).float().cuda()

    def forward(self, x, y):
        e = torch.matmul(x.unsqueeze(-2), self.labels)
        varLoss = torch.matmul(x.unsqueeze(-2), (self.labels - e) ** 2).squeeze(-1)
        meanLoss = (e.squeeze(-1) - y) ** 2
        return torch.mean(meanLoss), torch.mean(varLoss)


class MeanVarianceNormalizer(nn.Module):
    def __init__(self, mean, std):
        super(MeanVarianceNormalizer, self).__init__()
        self.mean = torch.from_numpy(
            np.array(mean).astype('float32')
        ).cuda()
        self.std = torch.from_numpy(
            np.array(std).astype('float32')
        ).cuda()

    def forward(self, x):
        x = x.squeeze(-1) * self.std + self.mean
        return x

"""-------  -------"""
class regressionClassification(nn.Module):
    def __init__(self, nClass):
        super(regressionClassification, self).__init__()
        self.nClass = nClass

        self.conv = nn.Sequential(
            BasicConv2d(57, 64, 3, bn=True, padding=1),
            BasicConv2d(64, 64, 3, bn=True, padding=1),
            BasicConv2d(64, 128, 3, bn=True, padding=1),
            BasicConv2d(128, 128, 3, bn=True, padding=1),
        )

        self.downsample = nn.Sequential(
            BasicConv2d(57, 128, 1, bn=True, relu=False, padding=0)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(128 * 1 * 1, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, self.nClass),
        )

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv(x)
        x = x + residual
        x = self.pool(x)
        x = self.fc(x)
        return x

"""------- classify for rain and non-rain -------"""
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

"""------- single base classifer for ordinal-------"""
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



"""-----Ordinal regression model-----------"""
class OrdinalRegressionModel(nn.Module):
    def __init__(self, nClass):
        super(OrdinalRegressionModel, self).__init__()
        self.nClass = nClass
        self.boosting = nn.ModuleList()

        """----------------Training nClass classifiers----------------""" 
        for i in range(self.nClass):
            oneClassifier = BaseClassifier()
            self.boosting.append(oneClassifier)

    def forward(self, x):
        
        # list sigmoid outputs from all classifers
        outputs = [self.boosting[i](x) for i in range(self.nClass)]
        # list → Tensor (torch.Size([1, nClass])
        return torch.cat(outputs, dim = 1)

    
"""------- Noisy encoder-decoder model-------"""
class AutoencoderBN(nn.Module):
    def __init__(self):
        super(AutoencoderBN, self).__init__()

        self.encoder = nn.Sequential(
            BasicConv2d(57, 32, 1, bn = True),
            BasicConv2d(32, 32, 3, bn = True, padding = 1),
        )
        
        self.encoderAfterNoise = nn.Sequential(
            nn.MaxPool2d(2),
            # -------------------------------------
            BasicConv2d(32, 64, 3, bn = True, padding = 1),
            BasicConv2d(64, 64, 3, bn = True, padding = 1),
        )

        self.decoder = nn.Sequential(
            BasicConv2d(64, 32, 3, bn = True, padding = 1),
            # upsample and output size → C × 29 × 29 
            nn.Upsample(size = (29, 29), mode ='bilinear',align_corners = True),
            BasicConv2d(32, 32, 3, bn = True, padding = 1),
            
            BasicConv2d(32, 57, 1, bn = True),
        )

    def forward(self, x):
        encoder = self.encoder(x)
        encoder = self.encoderAfterNoise(encoder)
        decoder = self.decoder(encoder)
        return encoder, decoder
    
    
    
    
    
    
    
    
    
    
    


class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()

        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(0.2),
            nn.Linear(57*19*19,1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024,128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128,1)
        )

    def forward(self,x):
        x = self.fc(x)
        return x


class SimpleRegressionModel(nn.Module):
    def __init__(self, nClass):
        super(SimpleRegressionModel, self).__init__()
        self.conv = nn.Sequential(
            BasicConv2d(64, 128, 3, padding=1),
            BasicConv2d(128, 128, 3, padding=1),
            BasicConv2d(128, 32, 1),
            BasicConv2d(32, 32, 1),
        )

        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))


        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32,1)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.fc(x)
        return x   
    
    
    
    
    
    
    
class Regression(nn.Module):
    def __init__(self, nClass):
        super(Regression, self).__init__()
        self.nClass = nClass

        self.subreg_l1 = nn.Sequential(

            BasicConv2d(64, 16, 1),
            BasicConv2d(16, 4, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),

        )

        self.subreg_l2 = nn.Sequential(
            BasicConv2d(64, 16, 1),
            BasicConv2d(16, 4, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.share_conv_l3_l4 = nn.Sequential(
            BasicConv2d(64, 64, 3, bn=True, padding=1),
        )

        self.subreg_l3 = nn.Sequential(
            BasicConv2d(64, 32, 1),
            BasicConv2d(32, 16, 1),
            BasicConv2d(16, 4, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))

        )

        self.subreg_l4 = nn.Sequential(
            BasicConv2d(64, 32, 1),
            BasicConv2d(32, 16, 1),
            BasicConv2d(16, 4, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))

        )

        self.subreg_l5 = nn.Sequential(
            BasicConv2d(64, 16, 1),
            BasicConv2d(16, 4, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )



        self.multiFc = nn.ModuleList([
            self.subreg_l1,
            self.subreg_l2,
            self.subreg_l5,
            nn.Sequential(
                self.share_conv_l3_l4,
                self.subreg_l3,
            ),
            nn.Sequential(
                self.share_conv_l3_l4,
                self.subreg_l4,
            )
        ])

    def forward(self, x):
        batchSize = x.size()[0]
        outputs = [torch.mean(self.multiFc[i](x), dim=(1, 2, 3)).unsqueeze(-1) for i in range(len(self.multiFc))]
        regressionValues = torch.cat(outputs, dim=1).view(batchSize, len(self.multiFc), 1)
        return regressionValues    


