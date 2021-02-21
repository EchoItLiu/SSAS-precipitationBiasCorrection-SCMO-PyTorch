import os
import torch
import sys
import json
import random
import torch.distributed as dist
import numpy as np
import os.path as osp
import math

def get_file_name(x):
    _get_file_name = lambda x: osp.splitext(osp.split(x)[1])[0]
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return np.array([_get_file_name(y) for y in x])
    else:
        return _get_file_name(x)
    
    
def featuresNorm(args, datasetName, utilPath, flag):
    datasetRoot = osp.join(args.basePath,
                              datasetName,
                              utilPath)
    if flag=='modal_ST':
        meanFilePath = osp.join(datasetRoot, "mean_st_modal.npy")
        stdFilePath = osp.join(datasetRoot, "std_st_modal.npy")        
    if flag=='modal_S':
        meanFilePath = osp.join(datasetRoot, "mean_s_modal.npy")
        stdFilePath = osp.join(datasetRoot, "std_s_modal.npy")
        print ('meanFilePath:', meanFilePath)
        
    if flag=='OBA':    
        meanFilePath = osp.join(datasetRoot, "mean_29.npy")
        stdFilePath = osp.join(datasetRoot, "std_29.npy")
        
    if flag=='ycluo':
        meanFilePath = '/mnt/pami14/yqliu/dataset_meteo/mean.npy'
        stdFilePath = '/mnt/pami14/yqliu/dataset_meteo/std.npy'
        
    if flag=='ycluo2':
        meanFilePath = '/mnt/pami14/yqliu/dataset_meteo/mean.npy'
        stdFilePath = '/mnt/pami14/yqliu/dataset_meteo/std.npy'
        
    if flag=='ycluo37':
        meanFilePath = '/mnt/pami14/yqliu/EC-BiasCorrecting/SHO/util_dataset/mean_st_modal_37.npy'
        stdFilePath = '/mnt/pami14/yqliu/EC-BiasCorrecting/SHO/util_dataset/std_st_modal_37.npy'        

         
    return np.load(meanFilePath), np.load(stdFilePath)


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is not None:
        rt /= world_size
    return rt


def save_osConfig(args, hyperPath):
    if not os.path.exists(hyperPath):
        os.makedirs(hyperPath)
            
    with open(os.path.join(hyperPath, "para.json"), "w") as f:
        json.dump(args.__dict__, f)
        

def generateOneHot(softmax):
    # Return the indices of the maximum values of a tensor across a dimension [N]
    maxIdxs = torch.argmax(softmax, dim = 1, keepdim = True).cpu().long()
    # init oneHot mask Tensor  [N × 2]
    oneHotMask = torch.zeros(softmax.shape, dtype = torch.float32)
    # convert "one-hot" to corresponding dimension(1) one-hot[N × 2] → [N × 1 × 2]
    oneHotMask = oneHotMask.scatter_(1, maxIdxs, 1.0)
    oneHotMask = oneHotMask.unsqueeze(-2)
    return oneHotMask


def reduce_tensor(tensor, world_size = None):
    # 复制一份(有梯度)
    rt = tensor.clone()
    dist.all_reduce(rt, op = dist.ReduceOp.SUM)
    if world_size is not None:
        rt /= world_size
    return rt

def generateMeanNoisy(batchSize, channel, sequence, hw, flag):
    if flag=='3D':
        noiseMean = torch.zeros(batchSize, channel, hw, hw)
    if flag=='4D':
        noiseMean = torch.zeros(batchSize, channel, sequence, hw, hw)
    
    return noiseMean

# 注意self方法不可以代到里面，否则一定会被重新赋值
def cropOneScale4EC(miniBatch, scale, EC_center):
    # crop ND * C * H * W →  ND * Hs * Ws
    curr_loc_l = EC_center - ceilHalf(scale)
    curr_loc_r = EC_center + ceilHalf(scale)
    curr_loc_t = EC_center + ceilHalf(scale)
    curr_loc_b = EC_center - ceilHalf(scale)
    ## 
    currscale_crop_miniBatch = miniBatch[:, :, curr_loc_l:(curr_loc_r+1), curr_loc_b:(curr_loc_t+1)]
    return currscale_crop_miniBatch

def cropOneLength4EC(miniBatch, curLength):
    ## N * C * D_curlength * H * W
    currlength_crop_miniBatch = miniBatch[:, :, curLength:, :, :]
    return currlength_crop_miniBatch

def cropOneScale4Sample(station_sample, scale, EC_center):
    # crop C * H * W →  1 * C * Hs * Ws
    curr_loc_l = EC_center - ceilHalf(scale)
    curr_loc_r = EC_center + ceilHalf(scale)
    curr_loc_t = EC_center + ceilHalf(scale)
    curr_loc_b = EC_center - ceilHalf(scale)
    ## 
    currscale_crop_sample = station_sample[:, curr_loc_l:(curr_loc_r+1), curr_loc_b:(curr_loc_t+1)].unsqueeze(0)
    return currscale_crop_sample

# 返回最优空间尺度
def calOptimScale4Sample(scale_losses, scale_stases, scale_range, station_sample, uniscale, EC_center, flag):
    if flag=='optim':
        ## require optim scale
        min_loss = min(scale_losses)
        min_loss_idx = [idx for idx, scale_loss in enumerate(scale_losses) if scale_loss==min_loss]
        optim_scale = scale_range[min_loss_idx[0]]
        optim_stas = scale_stases[min_loss_idx[0]]
        return optim_stas, optim_scale
          
    if flag=='uni':   
        optim_stas = cropOneScale4Sample(station_sample, uniscale, EC_center)
        return optim_stas

# 返回最优时间尺度
def calOptimTS4Sample(ts_losses, ts_stases, ts_range):
        ## require optim scale
    min_loss = min(ts_losses)
    min_loss_idx = [idx for idx, ts_loss in enumerate(ts_losses) if ts_loss==min_loss]
    optim_ts = ts_range[min_loss_idx[0]]
    optim_tsf = ts_stases[min_loss_idx[0]]
    return optim_ts, optim_tsf
           
def ceilHalf(x):
    ceils = math.ceil((x - 1) / 2)
    return ceils
      
def floorHalf(x):
    floors = math.floor((x - 1) / 2)
    return floors
    
