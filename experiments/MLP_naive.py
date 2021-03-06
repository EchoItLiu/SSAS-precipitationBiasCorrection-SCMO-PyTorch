import os
import os.path as osp
from torch.utils import data
import torch
import torch.nn as nn
from models.CompModels import _MLP
import numpy as np
import pickle
import tqdm
import random
import sys
import torch.distributed as dist
from utils.util_benchmarks import cal_PMAE, cal_TS_score, cal_MAE, filePathGenerate, dataGenerate, data_interplote_nor_gt, floorHalf
from torch.utils.tensorboard import SummaryWriter


class MLP(object):
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
                 activation,
                 solver,
                 alpha,
                 batch_size,
                 learning_rate,
                 learning_rate_init,
                 shuffle,
                 max_iter,
                 warm_start, 
                 early_stopping, 
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
        #
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init        
        self.shuffle = shuffle        
        self.warm_start = warm_start
        self.early_stopping = early_stopping        
        self.max_iter = max_iter
        #
        self.EC_center = floorHalf(self.ECSize)
        #
        self.writer = SummaryWriter(osp.join(self.args.save_root, 'tensorboard'))
        self.models_save_dir = osp.join(self.args.save_root, 'save_models') 
        os.makedirs(self.models_save_dir, exist_ok=True)        
        self.benchmark_name = benchmark_name
        self.repetitions = repetitions
                       

    def __repr__(self):
        return self.__class__.__name__

    def init_module(self):
      
          # ERA_Benchmark.py & CompModels.py
        self.mlp = _MLP(
                self.activation,
                self.solver,
                self.alpha,
                self.batch_size,
                self.learning_rate,
                self.learning_rate_init,     
                self.shuffle,        
                self.warm_start,
                self.early_stopping,        
                self.max_iter) 
        # data
        totalGribFileList = filePathGenerate(self.dataset_path, self.startDate, self.endDate)
       # Dataset split
        trainCropFileList, testCropFileList = dataGenerate(totalGribFileList, self.split_ratio)             # Trainset  
        self.featuresArr_train, self.labelsVec_train = data_interplote_nor_gt(trainCropFileList, self.EC_center, self.crop_scale, self.benchmark_name)
        # Testset
        self.featuresArr_test, self.labelsVec_test = data_interplote_nor_gt(testCropFileList, self.EC_center, self.crop_scale, self.benchmark_name)        

    def init_model(self):
        self.mlp.init_model.fit(self.featuresArr_train, self.labelsVec_train)
                    
    def fit(self):
        mae_best = 100000
        for repeat_num in range(self.repetitions): 
            # fit
            self.init_model()
            #
            preds_test = self.mlp.init_model.predict(self.featuresArr_test)
            #
            mae = cal_MAE(preds_test, self.labelsVec_test)
            pMae = cal_PMAE(preds_test, self.labelsVec_test) 
            #
            ts_01 = cal_TS_score(preds_test, self.labelsVec_test, 0.1)
            ts_1 = cal_TS_score(preds_test, self.labelsVec_test, 1)
            ts_10 = cal_TS_score(preds_test, self.labelsVec_test, 10)

            info = {"MAE": mae,
                    "MPAE": pMae,
                    "Ts0.1": ts_01,
                    "Ts1:": ts_1,
                    "Ts10:": ts_10,
                    }
            print("======= MLP Results Show =======".format(repeat_num + 1))
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

            if mae < mae_best:
                  
                # return dictionary
                dict_params_mlp = self.mlp.init_model.get_params()
                f = open(os.path.join(self.models_save_dir, 'mlp_{}'.format(str(repeat_num))),'w')
                # 必须是string类型才能保存，读取用eval()
                f.write(str(dict_params_mlp))
                f.close()
        
