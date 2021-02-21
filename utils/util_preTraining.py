import os
import torch
import torch.nn as nn
import sys
import json
import random
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import os.path as osp
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
klLoss = nn.KLDivLoss()

def flagMeBatch4Label(features, seqLabel, meFlag):
    ## update torch.Tensor
    feature_cha = features.size(1)
    hw = features.size(3)
    #N * C * (D-1) * H * W → N * (D-1) * C * H * W → N(D-1) * C * H * W
    features_up = features.transpose(1,2).contiguous().view(-1, feature_cha, hw, hw)
    # wind
    if meFlag=='wind':
        # N * D * 2 → ND * 2 
        seqLabel_up = seqLabel.contiguous().view(-1,2)
        seqLabel_windd = seqLabel_up[:,0]
        seqLabel_winds = seqLabel_up[:,1]
        count_9999_wind = 0
#         for idx in range(len(seqLabel_windd)):
#             if seqLabel_windd[idx]==9999 and seqLabel_winds[idx]==9999:
#                 print ('--9999wind--')
#                 count_9999_wind = count_9999_wind +1
#         print ('count_9999_wind:', count_9999_wind)
        
        # binary flag list [ND] 1为非9999和0样本
        ND1_labelFlags = [1 if seqLabel_windd[idx]!=0 and seqLabel_winds[idx]!=0 and seqLabel_windd[idx]!=9999 and seqLabel_winds[idx]!=9999 else 0 for idx in range(len(seqLabel_windd))]          
    else:
        # N * (D-1) → N(D-1)
        seqLabel_up = seqLabel.contiguous().view(-1)
        count_9999_other = 0
#         for idx, label in enumerate(seqLabel_up):
#             if label==9999.0:
#                 print ('--9999other--')
#                 count_9999_other = count_9999_other +1
                
#         print ('count_9999_other:', count_9999_other)
      
        ND1_labelFlags = [1 if label!=0 and label!=9999 else 0 for idx, label in enumerate(seqLabel_up)]
        
    return features_up.to(device), seqLabel_up.to(device), ND1_labelFlags
 
def generateProbs4t1Label(t1Label, meRanges, meFlag):
    # wind
    if meFlag=='wind':
        # N * 2 → N 
        seqLabel_windd = t1Label[:,0]
        seqLabel_winds = t1Label[:,1]
        # binary flag list [ND] 1为非9999和0样本
        N2_labelFlags = [1 if seqLabel_windd[idx]!=0 and seqLabel_winds[idx]!=0 and seqLabel_windd[idx]!=9999 and seqLabel_winds[idx]!=9999 else 0 for idx in range(len(seqLabel_windd))]          
    else:
        # N → N
        N2_labelFlags = [1 if label!=0 and label!=9999 else 0 for idx, label in enumerate(t1Label)]
        
    if meFlag=='wind':
        Probs_up_wd = generateProbsLabels(t1Label[:,0], meRanges[0])
        Probs_up_ws = generateProbsLabels(t1Label[:,1], meRanges[1])     
        # Tuple: N1*len(range)*2
        Probs_up = (Probs_up_wd, Probs_up_ws)
        
    else:
        # N1*len(range)
        Probs_up = generateProbsLabels(t1Label, meRanges)        
        
    return t1Label.to(device), Probs_up, N2_labelFlags
        
def interFlagIndice(temGTFlags, pressGTFlags, windGTFlags, dewGTFlags, rainGTFlags):
    GT_label_flag = [temGTFlags[i]+pressGTFlags[i]+windGTFlags[i]+dewGTFlags[i]+rainGTFlags[i] for i in range(len(temGTFlags))]
    return GT_label_flag
     
def updateMeBatch4Label(features, seqLabel, dropRate, meFlag):
    ## update torch.Tensor
    feature_cha = features.size(1)
    hw = features.size(3)
    # N * C * D * H * W → N * D * C *  H * W → ND * C * H * W
    features_v = features.transpose(1,2).contiguous().view(-1, feature_cha, hw, hw)
    ## wind
    if meFlag=='wind':
        # N * D * 2 → ND * 2 
        seqLabel_v = seqLabel.view(-1,2)
        seqLabel_windd = seqLabel_v[:,0]
        seqLabel_winds = seqLabel_v[:,1]
        unmask_label_ids = [idx for idx in range(len(seqLabel_windd)) if seqLabel_windd[idx]!=0 and seqLabel_winds[idx]!=0 and seqLabel_windd[idx]!=9999 and seqLabel_winds[idx]!=9999]
        
    else:
        seqLabel_v = seqLabel.view(-1)
        unmask_label_ids = [idx for idx, label in enumerate(seqLabel_v) if label!=0 and label!=9999]
        
    # N1D * C * H * W
    features_up = features_v[unmask_label_ids]
    # N1D/ N1D * 2
    seqLabel_up = seqLabel_v[unmask_label_ids]
    ## view and crop randomly 
    view_len = int(features_up.size(0) * (1-dropRate))
#     print ('view_len:', view_len)  
    features_up = features_up[0:(view_len+1)]
    seqLabel_up = seqLabel_up[0:(view_len+1)]
    return features_up.to(device), seqLabel_up.to(device)


def updateTSMeBatch4Label(features, targets, meRanges, dropRate, meFlag):
    ## update torch.Tensor
    feature_cha = features.size(1)
    hw = features.size(3)
    # N * C * D * H * W → N * D * C *  H * W → ND * C * H * W
    features_v = features.transpose(1,2).contiguous().view(-1, feature_cha, hw, hw)
    ## wind
    if meFlag=='wind':
        # N * 2  
        Target_windd = targets[:,0]
        Target_winds = targets[:,1]
        unmask_label_ids = [idx for idx in range(len(Target_windd)) if Target_windd[idx]!=0 and Target_windd[idx]!=0 and Target_winds[idx]!=9999 and Target_winds[idx]!=9999] 
        
    else:
        unmask_label_ids = [idx for idx, target in enumerate(targets) if target!=0 and target!=9999]
        
    # N1 * C * D * H * W → N1 / N1*2
    features_up = features[unmask_label_ids]
    # N1/ N1 * 2
    targets_up = targets[unmask_label_ids]
    ## view and crop randomly 
    view_len = int(features_up.size(0) * (1-dropRate))
#     print ('view_len:', view_len)  
    features_up = features_up[0:(view_len+1)]
    targets_up = targets_up[0:(view_len+1)]
    ## generate probs labels N
    
    if meFlag=='wind':
        Probs_up_wd = generateProbsLabels(targets_up[:,0], meRanges[0])
        Probs_up_ws = generateProbsLabels(targets_up[:,1], meRanges[1])
        # Tuple: N1*len(range)*2
        Probs_up = (Probs_up_wd, Probs_up_ws)
        
    else:
        # N1*len(range)
        Probs_up = generateProbsLabels(targets_up, meRanges)
        
    return features_up.to(device), targets_up.to(device), Probs_up

'''
????打label结果有问题????

不建议学全0的，就算温度有0,问问储海
注:负值也可以用两点法表达出来
注: 注意小数点是否能表达

'''
def generateProbsLabels(targets, meRanges): 
    # [....,threshold] interval
    k = meRanges[1] - meRanges[0]
    # numpy
    targets = targets.numpy()
    probs = []    
    #
    for y in targets: 
        index = 0
        # init prob
        prob = np.zeros(len(meRanges))
        # upper Bound [0...,1]
        if  y > meRanges[-1]:
            prob[-1] = 1.
        # lower Bound [0,...,y/k-1,...,0]
        elif y < meRanges[0]:
            prob[0] = 1.
        # fall linesegment
        elif y % k == 0:
            prob[int(y/k) - 1] = 1
        # non-linesegment
        else: 
            # 得到区间下标
            for i in range(len(meRanges)):
                if meRanges[i] > y:
                    index = i
                    break 
            # 计算λ1 λ2       
            prob[index - 1] = 1 - (y - np.floor(y / k) * k)/k
            prob[index] = 1 - (np.ceil(y / k) * k - y)/k
        # [meRanges,...]     
        probs.append(prob)
        # N * len(meRanges)
    probs = torch.from_numpy(np.stack(probs).astype(np.float32))
    return probs.to(device)


def BP_ME(out, gt, optimizer, meFlag):
    # same dimension
    gt = gt.unsqueeze(-1)
    if meFlag=='wind':
        loss0 = F.mse_loss(out[:,0], gt[:,0])
        loss1 = F.mse_loss(out[:,1], gt[:,1])
        loss = 0.6 * loss0 + 0.4 * loss1 
    else:
        loss = F.mse_loss(out, gt)
    loss.backward()
    optimizer.step()
    print('Train-{}-Loss-{}'.format(meFlag, loss))
    
    
def BP_ME_TS(out_value, out_prob, gt_value, gt_prob, optimizer, meFlag):
    # same dimension
    gt_value = gt_value.unsqueeze(-1)
    
    if meFlag=='wind':
        loss0_mse = F.mse_loss(out_value[0], gt_value[:,0])
        loss1_mse = F.mse_loss(out_value[1], gt_value[:,1])
        loss_mse = 0.6*loss0_mse + 0.4*loss1_mse
        #
        
        loss0_kl = klLoss(out_prob[0], gt_prob[0])
        loss1_kl = klLoss(out_prob[1], gt_prob[1]) 
        loss_kl = 0.6*loss0_kl + 0.4*loss1_kl
        #
        loss_total = 0.5*loss_mse + 0.5*loss_kl 
        
    else:
        loss_mse = F.mse_loss(out_value, gt_value)
                
#         print ('out_prob:', out_prob)
#         print ('gt_prob:', gt_prob)
                
        loss_kl = klLoss(out_prob, gt_prob)
#         print ('loss_kl:', loss_kl)
        loss_total = 0.5*loss_mse + 0.5*loss_kl
             
    loss_total.backward()
    optimizer.step()
    print('Train-{}-Loss-{}'.format(meFlag, loss_total))
    
    
def load_preTraining_MEs(preTrainingPath, tem_ME, press_ME, wind_ME, dew_ME, rain_ME, flag):
    # 
    if flag=='MSM':
        tem_path = osp.join(preTrainingPath, 'tem', 'pretrained-MSM-tem.pth')
        press_path = osp.join(preTrainingPath, 'press', 'pretrained-MSM-press.pth')
        wind_path = osp.join(preTrainingPath, 'wind', 'pretrained-MSM-wind.pth')
        dew_path = osp.join(preTrainingPath, 'dew', 'pretrained-MSM-dew.pth')
        rain_path = osp.join(preTrainingPath, 'rain', 'pretrained-MSM-rain.pth')
    #    
    if flag=='MTM':
        tem_path = osp.join(preTrainingPath, 'tem', 'pretrained-MTM-tem.pth')
        press_path = osp.join(preTrainingPath, 'press', 'pretrained-MTM-press.pth')
        wind_path = osp.join(preTrainingPath, 'wind', 'pretrained-MTM-wind.pth')
        dew_path = osp.join(preTrainingPath, 'dew', 'pretrained-MTM-dew.pth')
        rain_path = osp.join(preTrainingPath, 'rain', 'pretrained-MTM-rain.pth')        
    # load module.paramsDict → remove '.module'
    tem_paramsDict = removeRedKey(torch.load(tem_path, map_location = 'cpu'))
    press_paramsDict = removeRedKey(torch.load(press_path, map_location = 'cpu'))
    wind_paramsDict = removeRedKey(torch.load(wind_path, map_location = 'cpu'))
    dew_paramsDict = removeRedKey(torch.load(dew_path, map_location = 'cpu'))
    rain_paramsDict = removeRedKey(torch.load(rain_path, map_location = 'cpu'))
     
    # load
    tem_ME.load_state_dict(tem_paramsDict) 
    press_ME.load_state_dict(press_paramsDict) 
    wind_ME.load_state_dict(wind_paramsDict) 
    dew_ME.load_state_dict(dew_paramsDict) 
    rain_ME.load_state_dict(rain_paramsDict) 
    return tem_ME, press_ME, wind_ME, dew_ME, rain_ME  
    
def removeRedKey(moduleParamsDict):
    new_state_dict = OrderedDict()
    for k, v in moduleParamsDict.items():
        name = k[7:] 
        new_state_dict[name] = v
    return new_state_dict

