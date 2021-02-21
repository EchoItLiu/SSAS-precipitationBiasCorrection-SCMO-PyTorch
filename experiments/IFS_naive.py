import os
import os.path as osp
from torch.utils import data
import torch
import torch.nn as nn
import numpy as np
import pickle
import tqdm
import random
import sys
import torch.distributed as dist
from utils.util_benchmarks import cal_PMAE, cal_TS_score, cal_MAE, filePathGenerate, dataGenerate, data_interplote_nor_gt, floorHalf
from torch.utils.tensorboard import SummaryWriter

class IFS(object):
    def __init__(self,
                 args,
                 dataset_type,
                 dataset_util,
                 startDate,
                 endDate,
                 split_ratio,
                 crop_scale,
                 ####                 
                 EC_size,
                 copy_X,
                 n_jobs,
                 repetitions,
                 benchmark_name,
                 **kwargs
                 ):
        ##        
        self.args = args
        self.dataset_name = args.dataset_name
        self.dataset_type = dataset_type
        self.dataset_path = osp.join(args.basePath, args.dataset_name, dataset_type)
        self.dataset_util = dataset_util
        self.startDate = startDate
        self.endDate = endDate
        self.ECSize = EC_size
        self.split_ratio = split_ratio 
        self.crop_scale = crop_scale
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        #
        self.EC_center = floorHalf(self.ECSize)
        #
        self.writer = SummaryWriter(osp.join(self.args.save_root, 'tensorboard'))
        self.models_save_dir = osp.join(self.args.save_root, 'save_models')        
        self.benchmark_name = benchmark_name
        self.repetitions = repetitions
                       

    def __repr__(self):
        return self.__class__.__name__

    def init_module(self):      
        self.totalGribFileList = filePathGenerate(self.dataset_path, self.startDate, self.endDate)

    def init_model(self):
        #
        trainCropFileList, testCropFileList = dataGenerate(self.totalGribFileList, self.split_ratio)                 #
        self.featuresArr, self.labelsVec = data_interplote_nor_gt(testCropFileList, self.EC_center, self.crop_scale, self.benchmark_name)
                    
    def fit(self):
        mae_best = 100000
        for repeat_num in range(self.repetitions):                                                    
            self.init_model()
            print ('featuresArr_shape:', self.featuresArr.shape)
            print ('featuresLabel_shape:', self.labelsVec.shape)
            
            mae = cal_MAE(self.featuresArr, self.labelsVec)
            pMae = cal_PMAE(self.featuresArr, self.labelsVec) 
            #
            ts_01 = cal_TS_score(self.featuresArr, self.labelsVec, 0.1)
            ts_1 = cal_TS_score(self.featuresArr, self.labelsVec, 1)
            ts_10 = cal_TS_score(self.featuresArr, self.labelsVec, 10)

            info = {"MAE": mae,
                    "MPAE": pMae,
                    "Ts0.1": ts_01,
                    "Ts1:": ts_1,
                    "Ts10:": ts_10,
                    }
            print("======= Precipitation Results on EC Integrated Forecast System Show =======".format(repeat_num + 1))
            print(info)                                                    
                                                                    
        # visual            
            self.writer.add_scalars(
                'Accuracy_{}'.format(self.benchmark_name), 
                {
                "MAE": mae,
                "MPAE": pMae,
                "Ts0.1": ts_01,
                "Ts1:": ts_1,
                "Ts10:": ts_10}, 
                repeat_num+1) 