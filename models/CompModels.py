import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from collections import Counter

class _SVR(object):
    def __init__(self, 
                _kernel,
                _degree, 
                _gamma, 
                _tol,
                _Penalty,
                _epsilon, 
                _shrinking, 
                _max_iter):
        super(_SVR, self).__init__()
        
        self.init_model = SVR(kernel=_kernel, degree=_degree, gamma=_gamma, tol=_tol, C=_Penalty, epsilon=_epsilon, shrinking=_shrinking, max_iter=_max_iter)
        
        
class _MLP(object):
    def __init__(self, 
        _activation,
        _solver,
        _alpha,
        _batch_size,
        _learning_rate,
        _learning_rate_init,        
        _shuffle,     
        _warm_start,
        _early_stopping,        
        _max_iter):
        super(_MLP, self).__init__()
        
        self.init_model = MLPRegressor(activation=_activation, solver=_solver, alpha=_alpha, batch_size=_batch_size, learning_rate=_learning_rate, learning_rate_init=_learning_rate_init, max_iter=_max_iter, shuffle=_shuffle, warm_start=_warm_start, early_stopping=_early_stopping)
        
        
class _Bayes(object):
    def __init__(self, 
                _priors,
                _rain_cls_interval 
                ):    
#     def __init__(self, 
#                 _alpha,
#                 _fit_prior,
#                 _rain_cls_interval,
#                 ):
        super(_Bayes, self).__init__()
        # cal prior class_num
        _class_prior = np.arange(0,90,_rain_cls_interval)
#         self.init_model = MultinomialNB(alpha=_alpha, fit_prior=_fit_prior, class_prior=None)
        self.init_model = GaussianNB(priors=None)
        

class _RF(object):
    def __init__(self, 
                 _n_estimators,
                 _criterion,
                 _max_features,
                 _min_samples_split,
                 _min_samples_leaf,
                 _min_weight_fraction_leaf,
                 _bootstrap,
                 _n_jobs,
                 _warm_start,
    ):
        super(_RF, self).__init__()
        
        self.init_model = RandomForestRegressor(n_estimators=_n_estimators, criterion=_criterion, max_depth=None, min_samples_split=_min_samples_split, min_samples_leaf=_min_samples_leaf, min_weight_fraction_leaf=_min_weight_fraction_leaf, max_features=_max_features, max_leaf_nodes=None, bootstrap=_bootstrap, n_jobs=_n_jobs, random_state=None, warm_start=_warm_start)        
        
        
class _FPN(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(_FPN, self).__init__()
        # 57 - 64
        self.smooth_s = BasicConv2d(in_channel, hidden_channel, 3, bn = True, padding=1)
        # 64 - 64 stride[1,1]
        self.c1 = nn.Sequential(
            bottleneck_block(hidden_channel, hidden_channel*2, 1),
            bottleneck_block(hidden_channel*2, hidden_channel, 1),
        )
        ## 瓶颈网络: 一大一小
        # 64 - 128 stride[2,1]
        self.c2 = nn.Sequential(
            bottleneck_block(hidden_channel, hidden_channel*4, 2),
            bottleneck_block(hidden_channel*4, hidden_channel*2, 1),
        )
        # 128 - 256 stride[2,1]
        self.c3 = nn.Sequential(
            bottleneck_block(hidden_channel*2, hidden_channel*8, 2),
            bottleneck_block(hidden_channel*8, hidden_channel*4, 1),
        )        
        # 256 - 512 stride[2,1]
        self.c4 = nn.Sequential(
            bottleneck_block(hidden_channel*4, hidden_channel*16, 2),
            bottleneck_block(hidden_channel*16, hidden_channel*8, 1),
        ) 
        #
        self.top = BasicConv2d(hidden_channel*8, hidden_channel*4, 3, bn = True, padding=1)
        self.latlayer1 = BasicConv2d(hidden_channel*4, hidden_channel*4, 1, bn = True)
        self.latlayer2 = BasicConv2d(hidden_channel*2, hidden_channel*4, 1, bn = True)
        self.latlayer3 = BasicConv2d(hidden_channel, hidden_channel*4, 1, bn = True)
        self.smooth_e = BasicConv2d(hidden_channel*4, hidden_channel, 3, bn = True, padding=1)
        #
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # fc
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(hidden_channel * 1 * 1, 32),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )
        
    def res(self, p, c):
        _,_,H,W = c.size()
        res_cp = c + F.upsample(p, size=(H,W), mode='bilinear')
        return res_cp
    
    def forward(self, x):
        # Bottom-up
        c1 = self.smooth_s(x) 
        c2 = self.c1(c1)
        c3 = self.c2(c2)
        c4 = self.c3(c3)
        c5 = self.c4(c4)
        # Top-down
        p5 = self.top(c5)
        # fuse and hidden-smooth 
        p4 = self.res(p5, self.latlayer1(c4))
        p3 = self.res(p4, self.latlayer2(c3))
        p2 = self.res(p3, self.latlayer3(c2))
        # Smooth
        p2 = self.smooth_e(p2)
        # pooling HW for p2
        p2 = self.pool(p2)
        # last layer in p
        pred = self.fc(p2)                   
        return pred
    
    
# 记得方法加nn.DataParallel
class _LSTM(nn.Module):
    def __init__(self,in_channel, hidden_channel):
        super(_LSTM, self).__init__()
        self.in_channel = in_channel        
        self.hidden_channel = hidden_channel
        self.hidden_channel_after = hidden_channel-128
        #
        self.lstm1 = nn.LSTMCell(in_channel, hidden_channel)        
        self.lstm2 = nn.LSTMCell(hidden_channel, self.hidden_channel_after)        
        #
#         self.dropout = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(self.hidden_channel_after, 64)
        self.linear2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # D*N*f
        seq_length = x.size(0)
        #
        h_t_1 = torch.zeros(x.size(1), self.hidden_channel).cuda()
        c_t_1 = torch.zeros(x.size(1), self.hidden_channel).cuda()
        h_t_2 = torch.zeros(x.size(1), self.hidden_channel_after).cuda()
        c_t_2 = torch.zeros(x.size(1), self.hidden_channel_after).cuda()            
        for i in range(seq_length):
            h_t_1, c_t_1 = self.lstm1(x[i], (h_t_1, c_t_1))
            h_t_2, c_t_2 = self.lstm2(h_t_1, (h_t_2, c_t_2))
            
        h21 = self.linear1(h_t_2)
        #
        y_t = self.linear2(h21).squeeze()

#         print ('y_t_size:', y_t.size())
        return y_t
                                 
class _ConvLSTM(nn.Module):
    def __init__(self, tsLength, inputChannelNums, hiddenChannelNums, ec_size, devices):
        super(_ConvLSTM, self).__init__()
        self._w_h = ec_size
        self.device = devices                
        # ts representation - stacked 2 layers
        self.encoderConvLSTM_layer1 = ConvLSTMCell(tsLength, inputChannelNums, hiddenChannelNums, ec_size, devices, True)
        #
        self.encoderConvLSTM_layer2 = ConvLSTMCell(tsLength, inputChannelNums, hiddenChannelNums, ec_size, devices, False)        
    def forward(self, x): 
        # ConvLSTM
        outputsL1, _ = self.encoderConvLSTM_layer1(x, states = None)
        _, ts_y_last = self.encoderConvLSTM_layer2(outputsL1, states = None)                             
        return ts_y_last
                          
class bottleneck_block(nn.Module):
    def __init__(self, in_channel, out_channel, curr_stride):
        super(bottleneck_block, self).__init__()        
        self.shortcut = BasicConv2d(in_channel, out_channel, 1, bn=True, stride=curr_stride)
        # Bottleneck
        self.bk = nn.Sequential(
            BasicConv2d(in_channel, 64, 1, bn=True),
            # change scale
            BasicConv2d(64, 64, 3, bn=True, stride=curr_stride, padding=1),
            BasicConv2d(64, out_channel, 1, bn=True),
            )
        
    def forward(self, x):
        res = self.shortcut(x)
        x = self.bk(x)
        b_x = x + res 
        return b_x
                                                             
# Convolutional Long Short-Term Memeory models
class ConvLSTMCell(nn.Module):
    def __init__(self, tsLength, inputChannelNums, hiddenChannelNums, ecsize, device, layer_flag):
        super(ConvLSTMCell, self).__init__()
        if layer_flag==True:
            self.ts_conv = nn.Conv2d(in_channels=(inputChannelNums + hiddenChannelNums),
                                     out_channels=hiddenChannelNums * 4,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
            
        else:
            self.ts_conv = nn.Conv2d(in_channels=hiddenChannelNums * 2,
                                     out_channels=hiddenChannelNums * 4,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)            
        
           
        self.ts_bn = nn.BatchNorm2d(hiddenChannelNums * 4)
        # smooth
        self._conv = nn.Sequential(
             nn.Conv2d(in_channels=hiddenChannelNums, out_channels=hiddenChannelNums, kernel_size=3, stride=1, padding = 1)
#             BasicConv2d(128, 128, 3,  padding = 1),
#             BasicConv2d(128, 32, 3, padding =1),
#             BasicConv2d(32, 32, 1),
        )
        
        self.ac = nn.ReLU(True)        
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(hiddenChannelNums, 1),
        )
        #
        self._state_height, self._state_width = ecsize, ecsize
        # LSTM W
        self.W_ci = nn.Parameter(torch.zeros(1, hiddenChannelNums, self._state_height, self._state_width))
        self.W_cf = nn.Parameter(torch.zeros(1, hiddenChannelNums, self._state_height, self._state_width))
        self.W_co = nn.Parameter(torch.zeros(1, hiddenChannelNums, self._state_height, self._state_width))
        # LSTM B
        self.b_i = nn.Parameter(torch.zeros(1, hiddenChannelNums, self._state_height, self._state_width))
        self.b_f = nn.Parameter(torch.zeros(1, hiddenChannelNums, self._state_height, self._state_width))
        self.b_c = nn.Parameter(torch.zeros(1, hiddenChannelNums, self._state_height, self._state_width))
        self.b_o = nn.Parameter(torch.zeros(1, hiddenChannelNums, self._state_height, self._state_width))
        #
        self._seq = tsLength 
        self._input_channel =  inputChannelNums
        self._hidden_channel_nums = hiddenChannelNums
        #
        self.device = device
        self.layer_flag = layer_flag
               
    # inputs: N * C * D * h * w  
    # num_filter is channel number of h 
    def forward(self, inputs = None, states = None):
        # D * N * C * h * w origin inputs        
        if self.layer_flag==True:
            inputs = inputs.transpose(0,2).transpose(1,2)
        # 
        else:   
            inputs = torch.stack(inputs)            

        if states is None:
            # LSTM Cell and hidden
            c = torch.zeros((inputs.size(1), self._hidden_channel_nums, self._state_height,
                                  self._state_width), dtype = torch.float).to(self.device)
            h = torch.zeros((inputs.size(1), self._hidden_channel_nums, self._state_height,
                             self._state_width), dtype = torch.float).to(self.device)
            
        else:
            h, c = states

        outputs = []
        #
        for index in range(self._seq):
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                      self._state_width), dtype=torch.float).to(self.device)
            else:
                x = inputs[index]
                
            st_x = torch.cat([x, h], dim = 1)
            #
            conved_st_x = self.ts_conv(st_x)
            conved_st_x_bn = self.ts_bn(conved_st_x)
            conv_i, conv_f, conv_c, conv_o = torch.chunk(conved_st_x_bn, 4, dim = 1)
            # save spatiotemporal features
            i = torch.sigmoid(conv_i + self.W_ci * c + self.b_i)
            f = torch.sigmoid(conv_f + self.W_cf * c + self.b_f)
            # c_{t-1} → c_t
            c = f * c + i * torch.tanh(conv_c + self.b_c)
            #
            o = torch.sigmoid(conv_o + self.W_co * c + self.b_o)
            h = o * torch.tanh(c)
            
            # output y_hat
            x_deep = self._conv(h)
            x_relu = self.ac(x_deep)
            x_pooling = self.pooling(x_relu)
            y_hat_TimeStamp_last = self.fc(x_pooling)
            #
            outputs.append(h)        
        # N * hiC * h * w (hidden in the last timeStamp) | N * 1                    
        return outputs, y_hat_TimeStamp_last
    
    
    
class _ConvLSTM(nn.Module):
    def __init__(self, tsLength, inputChannelNums, hiddenChannelNums, ec_size, devices):
        super(_ConvLSTM, self).__init__()
        self._w_h = ec_size
        self.device = devices                
        # ts representation - stacked 2 layers
        self.encoderConvLSTM_layer1 = ConvLSTMCell(tsLength, inputChannelNums, hiddenChannelNums, ec_size, devices, True)
        #
        self.encoderConvLSTM_layer2 = ConvLSTMCell(tsLength, inputChannelNums, hiddenChannelNums, ec_size, devices, False)        
    def forward(self, x): 
        # ConvLSTM
        outputsL1, _ = self.encoderConvLSTM_layer1(x, states = None)
        _, ts_y_last = self.encoderConvLSTM_layer2(outputsL1, states = None)                             
        return ts_y_last  


# class _TPN(nn.Module):
#     def __init__(self, tsLength, inputChannelNums, hiddenChannelNums, ec_size, devices):
#         super(_ConvLSTM, self).__init__()
#         self._w_h = ec_size
#         self.device = devices                
#         # ts representation - stacked 2 layers
#         pass
#     def forward(self, x): 
#         # ConvLSTM
#         outputsL1, _ = self.encoderConvLSTM_layer1(x, states = None)
    
    
#         return ts_y_last      
    
    

    
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