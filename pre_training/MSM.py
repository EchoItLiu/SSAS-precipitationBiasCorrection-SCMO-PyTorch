from __future__ import absolute_import, division

import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("/home/yqliu/EC-BiasCorrecting/pre_training")
from deformConv.layers import ConvOffset2D

'''
MSM  shortCut→Bottleneck→Dconv1→Dconv2→Dconv3→pooling→fc 因为是多模态数据(温压风湿降水)，语义信息会有辨别度，最好分5个模型实现比较好，而且模型都要有不同
'''

# class DeformConvNetI   
class DeformConvNetI(nn.Module):
    def __init__(self, in_channel):
        super(DeformConvNetI, self).__init__()
        
        # shallow feature map
        self.shortcut = BasicConv2d(in_channel, 256, 1, bn=True, padding=0)
        # Bottleneck
        self.bk = nn.Sequential(
            BasicConv2d(256, 64, 1, bn=True),
            BasicConv2d(64, 64, 3, bn=True, padding=1),
            BasicConv2d(64, 256, 1, bn=True),
            )
        # smooth
        self.smo = BasicConv2d(256, 32, 3, bn=True, padding=1)
        
        # Dconv1
        self.offset1 = ConvOffset2D(32)
        self.conv1 = BasicConv2d(32, 64, 3, bn=True, padding=1)

#         # Dconv2
#         self.offset2 = ConvOffset2D(64)
#         self.conv2 = BasicConv2d(64, 128, 3, bn=True, padding=1)

        # Dconv3
        self.offset3 = ConvOffset2D(64)
        self.conv3 = BasicConv2d(64, 128, 3, bn=True, padding=1)

        # Pooling
        self.pooling  = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # MLP
        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(0.5),            
            nn.Linear(128,32),
            nn.Linear(32, 1),
            )

    def forward(self, x):
        res = self.shortcut(x)
        x = self.bk(res) + res
        
        x = self.smo(x)
  
        x = self.offset1(x)
        x = self.conv1(x)
        
#         x = self.offset2(x)
#         x = self.conv2(x)
        
        x = self.offset3(x)
        x = self.conv3(x)        
        
        x = self.pooling(x)  
        x = self.fc(x)
        return x
    
# class DeformConvNetII        
class DeformConvNetII(nn.Module):
    def __init__(self, in_channel):
        super(DeformConvNetII, self).__init__()
        
        # shallow feature map
        self.shortcut = BasicConv2d(in_channel, 256, 1, bn=True, padding=0)
        # Bottleneck
        self.bk = nn.Sequential(
            BasicConv2d(256, 64, 1, bn=True),
            BasicConv2d(64, 64, 3, bn=True, padding=1),
            BasicConv2d(64, 256, 1, bn=True),
            )
        # smooth
        self.smo = BasicConv2d(256, 32, 3, bn=True, padding=1)
        
        # Dconv1
        self.offset1 = ConvOffset2D(32)
        self.conv1 = BasicConv2d(32, 64, 3, bn=True, padding=1)

        # Dconv2
#         self.offset2 = ConvOffset2D(64)
#         self.conv2 = BasicConv2d(64, 128, 3, bn=True, padding=1)

        # Dconv3
        self.offset3 = ConvOffset2D(64)
        self.conv3 = BasicConv2d(64, 128, 3, bn=True, padding=1)
        # Pooling
        self.pooling  = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # MLP branch (2 tasks)
        self.flat = Flatten()
        self.fc_dre = nn.Linear(128, 1)
        self.fc_voc = nn.Linear(128, 1)        
        

    def forward(self, x):
        res = self.shortcut(x)
        x = self.bk(res) + res
        
        x = self.smo(x)
  
        x = self.offset1(x)
        x = self.conv1(x)
        
#         x = self.offset2(x)
#         x = self.conv2(x)
        
        x = self.offset3(x)
        x = self.conv3(x)        
        
        x = self.pooling(x)
        x = self.flat(x)
        x_dre = self.fc_dre(x)
        x_voc = self.fc_voc(x)    
        x_total = torch.cat((x_dre, x_dre),-1) 
        return x_total    
          
# class DeformConvNetIII
pass



## util classes
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

    def forward(self, _input):
        return _input.view(_input.size(0), -1)

                                                                                    
#     def freeze(self, module_classes):
#         '''
#         freeze modules for finetuning
#         '''
#         for k, m in self._modules.items():
#             if any([type(m) == mc for mc in module_classes]):
#                 for param in m.parameters():
#                     param.requires_grad = False

#     def unfreeze(self, module_classes):
#         '''
#         unfreeze modules
#         '''
#         for k, m in self._modules.items():
#             if any([isinstance(m, mc) for mc in module_classes]):
#                 for param in m.parameters():
#                     param.requires_grad = True

#     def parameters(self):
#         return filter(lambda p: p.requires_grad, super(DeformConvNet, self).parameters()
                      
         
                                         
# def get_deform_cnn(trainable=True, freeze_filter=[nn.Conv2d, nn.Linear]):
#     model = DeformConvNet()
#     if not trainable:
#         model.freeze(freeze_filter)
#     return model