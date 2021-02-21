import os
import os.path as osp
from torch.utils import data
import torch
import torch.nn as nn
from models.ERA_Benchmark import _LR
import numpy as np
import pickle
import tqdm
import random
import sys
import torch.distributed as dist
from utils.util_benchmarks import cal_PMAE, cal_TS_score, cal_MAE, filePathGenerate, dataGenerate, data_interplote_nor_gt, iterData_ts_gt_nor, floorHalf
from torch.utils.tensorboard import SummaryWriter

class LR(object):
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
                 ts_seq,
                 in_channel,
                 fit_intercept,
                 normalize,
                 copy_X,
                 n_jobs,
                 LR_repetitions,
                 LR_iter_repetitions,
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
        self.ts_seq = ts_seq
        self.in_channel = in_channel
        self.split_ratio = split_ratio 
        self.crop_scale = crop_scale
        #
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        #
        self.EC_center = floorHalf(self.ECSize)
        #
        self.writer = SummaryWriter(osp.join(self.args.save_root, 'tensorboard'))
        self.models_save_dir = osp.join(self.args.save_root, 'save_models')
        os.makedirs(self.models_save_dir, exist_ok=True)        
        self.benchmark_name = benchmark_name
        self.LR_repetitions = LR_repetitions
        self.LR_iter_repetitions = LR_iter_repetitions
        
        
        #########
#         self.control = 'LR'
        self.control = 'iter_LR'
        #########
    

    def __repr__(self):
        return self.__class__.__name__

    def init_module(self):
        
        # data
        totalGribFileList = filePathGenerate(self.dataset_path, self.startDate, self.endDate)
       # Dataset split
        trainCropFileList, testCropFileList = dataGenerate(totalGribFileList, self.split_ratio)        
        
        if self.control=='LR':
            self.lr_naive = _LR(
                     self.fit_intercept,
                     self.normalize,
                     self.copy_X,
                     self.n_jobs,
                     'naive')
           # LR Trainset
            self.featuresArr_train, self.labelsVec_train = data_interplote_nor_gt(trainCropFileList, self.EC_center, self.crop_scale, self.benchmark_name)
            # LR Testset
            self.featuresArr_test, self.labelsVec_test = data_interplote_nor_gt(testCropFileList, self.EC_center, self.crop_scale, self.benchmark_name)  

        
        if self.control=='iter_LR':
            self.lr_iter = _LR(
                     self.fit_intercept,
                     self.normalize,
                     self.copy_X,
                     self.n_jobs,
                     'iter')         
            ##
            # LR iter Trainset  
            self.featuresNorArr_train, self.seqLabelsArr_train = iterData_ts_gt_nor(trainCropFileList,  self.crop_scale, self.EC_center, self.in_channel, self.ts_seq)
            # LR iter Testset
            self.featuresNorArr_test, self.seqLabelsArr_test = iterData_ts_gt_nor(testCropFileList, self.crop_scale, self.EC_center, self.in_channel, self.ts_seq)        

    def init_model(self):
        # fit
        if self.control=='LR':
            self.lr_naive.init_model_naive.fit(self.featuresArr_train, self.labelsVec_train)
            
        if self.control=='iter_LR':
            ## 只训练LR的分类能力
            for seq_id in range(self.ts_seq):
                print ('---')
                self.lr_iter.init_model_iter.fit(self.featuresNorArr_train[seq_id,:,:], self.seqLabelsArr_train[seq_id])
            
                       
    def fit(self):
        if self.control=='LR':
            mae_best = 100000
            for repeat_num in range(self.LR_repetitions):
                # fit
                self.init_model()
                # 
                preds_test = self.lr_naive.init_model_naive.predict(self.featuresArr_test)
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
                print("======= LR Results Show =======".format(repeat_num + 1))
                print(info)                                                    

            # visual            
#                 self.writer.add_scalars(
#                     'Accuracy_{}'.format(self.benchmark_name), 
#                     {
#                     "MAE": mae,
#                     "MPAE": pMae,
#                     "Ts0.1": ts_01,
#                     "Ts1:": ts_1,
#                     "Ts10:": ts_10}, 
#                     repeat_num+1) 

                if mae < mae_best:
                    # return dictionary
                    dict_params_lr_naive = self.lr_naive.init_model_naive.get_params()
                    f = open(os.path.join(self.models_save_dir, 'lr_model'),'w')
                    
#                     f = open(os.path.join(self.models_save_dir, 'lr_{}'.format(str(repeat_num))),'w')
                    # 必须是string类型才能保存，读取用eval()
                    f.write(str(dict_params_lr_naive))
                    f.close()
        
                             
        if self.control=='iter_LR':
            mae_best = 100000
            for repeat_num in range(self.LR_iter_repetitions):
                # fit
                self.init_model()
                # iterative prediction
                for seq_id in range(self.ts_seq):
                    if seq_id==0:
                        preds_test = self.lr_iter.init_model_iter.predict(self.featuresNorArr_test[seq_id,:,:])
                        
                    else:
                        preds_test = preds_test.reshape(-1,1)
                        # 删掉一个维度，否则对齐不上LR之前训练的参数大小
                        mix_inputs = np.concatenate((preds_test, self.featuresNorArr_test[seq_id,:,1:]), 1)
                        print ('mix_inputs_shape:', mix_inputs.shape)
                        preds_test = self.lr_iter.init_model_iter.predict(mix_inputs) 
                                           
                #
                mae = cal_MAE(preds_test, self.seqLabelsArr_test[-1])
                pMae = cal_PMAE(preds_test, self.seqLabelsArr_test[-1]) 
                #
                ts_01 = cal_TS_score(preds_test, self.seqLabelsArr_test[-1], 0.1)
                ts_1 = cal_TS_score(preds_test, self.seqLabelsArr_test[-1], 1)
                ts_10 = cal_TS_score(preds_test, self.seqLabelsArr_test[-1], 10)
                #             
                info = {"MAE": mae,
                        "MPAE": pMae,
                        "Ts0.1": ts_01,
                        "Ts1:": ts_1,
                        "Ts10:": ts_10,
                        }
                print("======= Iterative LR Results Show =======".format(repeat_num + 1))
                print(info)                                                    

            # visual            
                self.writer.add_scalars(
                    'Accuracy_{}'.format('iter_'+self.benchmark_name), 
                    {
                    "MAE": mae,
                    "MPAE": pMae,
                    "Ts0.1": ts_01,
                    "Ts1:": ts_1,
                    "Ts10:": ts_10}, 
                    repeat_num+1) 

                if mae < mae_best:
                    # return dictionary
                    dict_params_lr_iter = self.lr_iter.init_model_iter.get_params()
                    f = open(os.path.join(self.models_save_dir, 'lr_iter_{}'.format(str(repeat_num))),'w')
                    # 必须是string类型才能保存，读取用eval()
                    f.write(str(dict_params_lr_iter))
                    f.close()