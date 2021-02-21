import os
import os.path as osp
from torch.utils import data
import torch
import torch.nn as nn
from models.CompModels import _Bayes
import numpy as np
import pickle
import tqdm
import random
import sys
import torch.distributed as dist
from utils.util_benchmarks import cal_PMAE, cal_TS_score4Bayes, cal_CE, filePathGenerate, dataGenerate, data_interplote_nor_gt, floorHalf
from torch.utils.tensorboard import SummaryWriter

class Bayes(object):
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
                 rain_cls_interval,
                 alpha,
                 fit_prior,
                 benchmark_name,
                 repetitions,
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
        self.rain_cls_interval = rain_cls_interval
        self.alpha = alpha
        self.fit_prior = fit_prior
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
        # data
        totalGribFileList = filePathGenerate(self.dataset_path, self.startDate, self.endDate)
        # Dataset split
        trainCropFileList, testCropFileList = dataGenerate(totalGribFileList, self.split_ratio)                 # Trainset  
        self.featuresArr_train, self.labelsVec_train = data_interplote_nor_gt(trainCropFileList, self.EC_center, self.crop_scale, self.benchmark_name+str(self.rain_cls_interval))
#         self.n_class = len(np.unique(self.labelsVec_train))
        # Testset
        self.featuresArr_test, self.labelsVec_test = data_interplote_nor_gt(testCropFileList, self.EC_center, self.crop_scale, self.benchmark_name+str(self.rain_cls_interval)) 
        
        #
        self.bayes = _Bayes(
            None,
            self.rain_cls_interval 
        )          
#         self.bayes = _Bayes(
#             self.alpha,
#             self.fit_prior,
#             self.rain_cls_interval
#         )         

    def init_model(self):
        # fit
        self.bayes.init_model.fit(self.featuresArr_train, self.labelsVec_train)
                    
    def fit(self):
        ce_best = 100000
        for repeat_num in range(self.repetitions): 
            # fit
            self.init_model()
            # n*n_classes
            preds_probs = self.bayes.init_model.predict_log_proba(self.featuresArr_test)
            preds_cls = self.bayes.init_model.predict(self.featuresArr_test)
            # 
            print ('preds_probs:', preds_probs)
            ce = cal_CE(preds_probs, self.labelsVec_test)
            #
#             ts_01 = cal_TS_score4Bayes(preds_cls, self.labelsVec_test, 0)
#             ts_1 = cal_TS_score4Bayes(preds_cls, self.labelsVec_test, 1)
#             ts_10 = cal_TS_score4Bayes(preds_cls, self.labelsVec_test, 2)
            
            print ('sklearn-R-score:', self.bayes.init_model.score(self.featuresArr_test, self.labelsVec_test))

            info = {"Cross-Entropy": ce,
#                     "Ts0.1": ts_01,
#                     "Ts1:": ts_1,
#                     "Ts10:": ts_10,
                    }
            print("======= Bayesian Results Show =======".format(repeat_num + 1))
            print(info)                                                    
                                                                    
        # visual            
#             self.writer.add_scalars(
#                 'Accuracy_{}'.format(self.benchmark_name), 
#                 {
#                 "Cross-Entropy": ce,
#                 "Ts0.1": ts_01,
#                 "Ts1:": ts_1,
#                 "Ts10:": ts_10}, 
#                 repeat_num+1) 

#             if ce < ce_best:
                # return dictionary
            dict_params_bayes = self.bayes.init_model.get_params()
            f = open(os.path.join(self.models_save_dir, 'bayes'),'w')

#                 f = open(os.path.join(self.models_save_dir, 'bayes_{}'.format(str(repeat_num))),'w')
                # 必须是string类型才能保存，读取用eval()
            f.write(str(dict_params_bayes))
            f.close()
        
