'''
'''

import os
import os.path as osp
from torch.utils import data
import torch
import torch.nn as nn
from models.CompModels import _FPN
import numpy as np
from torchvision import transforms
import pickle
import tqdm
import random
import sys
import torch.distributed as dist
from utils.util_main import featuresNorm
from utils.util_benchmarks import cal_PMAE, cal_TS_score, cal_MAE, filePathGenerate, dataGenerate, deepModels_dataLoader, data_interplote_nor_gt, floorHalf, ycluo1_dataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class FPN(object):
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
                 in_feature,
                 hidden_feature,
                 ts_seq,
                 train_batch_size,
                 test_batch_size,
                 training_inv,
                 learning_rate,
                 weight_decay, 
                 max_epoch,
                 benchmark_name,
                 **kwargs
                 ):
        ##        
        self.args = args
        self.dataset_name = args.dataset_name
        self.dataset_type = dataset_type
#         self.dataset_path = osp.join(args.basePath, args.dataset_name, dataset_type)
        ##△ ycluo
        self.dataset_path = '/mnt/pami14/yqliu/dataset_meteo/cropGrib'
        self.dataset_util = dataset_util
        self.startDate = startDate
        self.endDate = endDate
        self.ECSize = EC_size
        self.split_ratio = split_ratio 
        self.crop_scale = crop_scale
        #
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.training_inv = training_inv
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # luo
        self.max_epoch = 10
        
        self.num_workers = args.num_workers 
        #
        self.EC_center = floorHalf(self.ECSize)
        #
        self.writer = SummaryWriter(osp.join(self.args.save_root, 'tensorboard'))
        self.models_save_dir = osp.join(self.args.save_root, 'save_models')
        os.makedirs(self.models_save_dir, exist_ok=True)
        self.benchmark_name = benchmark_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __repr__(self):
        return self.__class__.__name__

    def init_module(self):
#         featuresMean_st, featuresStd_st = featuresNorm(self.args, self.dataset_name, self.dataset_util, 'modal_S')
        ##△ ycluo
        featuresMean_st, featuresStd_st = featuresNorm(self.args, self.dataset_name, self.dataset_util, 'ycluo')
        # 3DTransformer
#         transformer_4D = transforms.Compose([
#         transforms.Normalize(mean=featuresMean_st, std=featuresStd_st)
#     ])
        ##△ ycluo
        transformer_luo = transforms.Compose([
        transforms.Normalize(mean=featuresMean_st, std=featuresStd_st)
        ])  
        
       # dataLoader
        totalGribFileList = filePathGenerate(self.dataset_path, self.startDate, self.endDate)
       # Dataset split
        trainCropFileList, testCropFileList = dataGenerate(totalGribFileList, self.split_ratio)         
#         DatasetFPN_train = deepModels_dataLoader(self.args,
#             trainCropFileList,
#             transformer_4D
#             )
        
        ##△ ycluo
        DatasetFPN_train = ycluo1_dataLoader(
            trainCropFileList,
            transformer_luo
            )        

        # Trainset N * C * D * H * W
        self.TrainDataLoader = data.DataLoader(
            DatasetFPN_train, 
            batch_size=self.train_batch_size,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            ) 

        DatasetFPN_test = ycluo1_dataLoader(
            testCropFileList,
            transformer_luo
            )
    
        # Testset N * C * D * H * W
        self.TestDataLoader = data.DataLoader(
            DatasetFPN_test, 
            batch_size=self.test_batch_size,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            )           
        
    def init_model(self):
        # init
        torch.cuda.set_device(self.args.device_ids[0])
#         fpn = _FPN(
#             self.in_feature,
#             self.hidden_feature
#         )

        ##△ ycluo
        fpn = _FPN(
            37,
            self.hidden_feature
        )         
        if len(self.args.device_ids) > 1:
            self.fpn = nn.DataParallel(fpn, device_ids=self.args.device_ids).cuda()    
        else:
            self.fpn = fpn.cuda()
        print(self.args) 
        self.optimizer = torch.optim.Adam(self.fpn.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)        
              
    def fit(self):
        self.init_model()
        self.epoch_iterations = 0
        #
        self.mae_best = []
        self.mape_best = []
        self.ts0 = []
        self.ts1 = []
        self.ts10 = []
        self.maeBest = 100000
        
        # if max_epoch
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train()
            if self.max_epoch % self.training_inv == 0:
                self.inference()
            self.epoch_iterations = self.epoch_iterations + 1 
        
        #
        MAE_FPN = np.mean(self.mae_best)
        MAPE_FPN = np.mean(self.mape_best)
        TS0_FPN = np.mean(self.ts0)
        TS1_FPN = np.mean(self.ts1)
        TS10_FPN = np.mean(self.ts10)

        MAE_FPN_STD = np.std(self.mae_best)
        MAPE_FPN_STD = np.std(self.mape_best)
        TS0_FPN_STD = np.std(self.ts0)
        TS1_FPN_STD = np.std(self.ts1)
        TS10_FPN_STD = np.std(self.ts10)        
        
        info_avg = {"MAE": MAE_FPN,
                "MAE_FPN_STD": MAE_FPN_STD,    
                "MPAE": MAPE_FPN,
                "MAPE_FPN_STD": MAPE_FPN_STD,    
                "Ts0.1": TS0_FPN,
                "Ts0.1_std": TS0_FPN_STD,
                "Ts1:": TS1_FPN,
                "Ts1_std": TS1_FPN_STD,    
                "Ts10:": TS10_FPN,
                "Ts10_std": TS10_FPN_STD,    
                }
        print("======= FPN Final Results Show =======")
        
        print("info_AVG:", info_avg)                                                    
            
        
    ########Train FPN###########            
    def train(self):
        self.fpn.train()
        for batch_idx, (data, target) in tqdm.tqdm(
            enumerate(self.TrainDataLoader), 
            total=len(self.TrainDataLoader), 
            desc='Train FPN epoch=%d' % self.epoch, 
            ncols=100, 
            leave=False):
            ## Data
#             data = data[:,:,-1,:,:].to(self.device)
#             data = data[:,:,-1,:,:].to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            preds = self.fpn(data)
            ## Loss                      
            Loss = F.mse_loss(preds, target)
            #
            Loss.backward()
            self.optimizer.step()             
            #
            self.writer.add_scalars('Loss_{}'.format("FPN"), {
                'fpn_loss': Loss.item()},
                self.epoch_iterations)      
        print ('epoch-{}-last_miniBatch-MSELoss:'.format(self.epoch), Loss.item())
                         
                         
    ########Evaluation FPN###########            
    def inference(self):
        self.fpn.eval() 
        batch_preds = []
        batch_gts = []        
        for batch_idx, (data, target) in tqdm.tqdm(
            enumerate(self.TestDataLoader), total = len(self.TestDataLoader),
            desc = 'Test FPN epoch=%d' % self.epoch, ncols = 80,
            leave=False): 
            
            ## Data
#             data = data[:,:,-1,:,:].to(device=self.device)
            
            target = target.to(device=self.device)
            
            # Inference
            with torch.no_grad():
                preds = self.fpn(data)
                Loss = F.mse_loss(preds, target).item()
                batch_preds.append(preds.cpu().numpy().squeeze())
                batch_gts.append(target.cpu().numpy().squeeze())                
                
                
        # list → numpy
        preds = np.concatenate(batch_preds)
        labelsVec = np.concatenate(batch_gts)             
        #
        mae = cal_MAE(preds, labelsVec)
        pMae = cal_PMAE(preds, labelsVec) 
        #
        ts_01  = cal_TS_score(preds, labelsVec, 0.1)
        ts_1 = cal_TS_score(preds, labelsVec, 1)
        ts_10 = cal_TS_score(preds, labelsVec, 10)

        #     
        
        info = {"MAE": round(mae,2),
            "MPAE": round(pMae,2),
            "Ts0.1":round(ts_01,2),
            "Ts1:": round(ts_1,2),
            "Ts10:": round(ts_10,2),
            }        
        
        print("======= FPN Results Show =======".format(self.epoch_iterations + 1))
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
            self.epoch_iterations + 1) 

        if self.epoch_iterations>=3 and mae < self.maeBest:
            # return dictionary
            self.maeBest = mae
            dict_params_fpn = self.fpn.state_dict()
            torch.save(dict_params_fpn, osp.join(self.models_save_dir, 'fpn-{}.pt'.format(self.epoch_iterations+1)))
            
            self.mae_best.append(mae)
            self.mape_best.append(pMae)
            self.ts0.append(ts_01)
            self.ts1.append(ts_1)
            self.ts10.append(ts_10)
