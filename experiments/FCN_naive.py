import os
import os.path as osp
from torch.utils import data
import torch
import torch.nn as nn
from models.ERA_Benchmark import _FCN
import numpy as np
from torchvision import transforms
import pickle
import tqdm
import random
import sys
import torch.distributed as dist
from utils.util_main import featuresNorm
from utils.util_benchmarks import cal_PMAE, cal_TS_score, cal_MAE, filePathGenerate, dataGenerate, deepModels_dataLoader, data_interplote_nor_gt, floorHalf
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class FCN(object):
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
                 in_feature,
                 hidden_feature,
                 hidden_iter_feature, 
                 train_batch_size,
                 test_batch_size,
                 training_inv,
                 learning_rate,
                 weight_decay, 
                 max_epoch,
                 benchmark_name1,
                 benchmark_name2,
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
        self.split_ratio = split_ratio 
        self.crop_scale = crop_scale
        #
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.hidden_iter_feature = hidden_iter_feature 
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.training_inv = training_inv
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.num_workers = args.num_workers 
        #
        self.EC_center = floorHalf(self.ECSize)
        #
        self.writer = SummaryWriter(osp.join(self.args.save_root, 'tensorboard'))
        self.models_save_dir = osp.join(self.args.save_root, 'save_models')
        os.makedirs(self.models_save_dir, exist_ok=True)        
        self.benchmark_name1 = benchmark_name1
        self.benchmark_name2 = benchmark_name2        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #########
#         self.control = 'FCN'
        self.control = 'iter_FCN'          
        #########

    def __repr__(self):
        return self.__class__.__name__

    def init_module(self):
        featuresMean_st, featuresStd_st = featuresNorm(self.args, self.dataset_name, self.dataset_util, 'modal_ST')
        # 4DTransformer
        transformer_4D = transforms.Compose([
        transforms.Normalize(mean=featuresMean_st, std=featuresStd_st)
    ])
        
       # dataLoader
        totalGribFileList = filePathGenerate(self.dataset_path, self.startDate, self.endDate)
       # Dataset split
        trainCropFileList, testCropFileList = dataGenerate(totalGribFileList, self.split_ratio)         
        DatasetFCN_train = deepModels_dataLoader(self.args,
            trainCropFileList,
            transformer_4D
            )
        # Trainset N * C * D * H * W
        self.TrainDataLoader = data.DataLoader(
            DatasetFCN_train, 
            batch_size=self.train_batch_size,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            )        
        #
        DatasetFCN_test = deepModels_dataLoader(self.args,
            testCropFileList,
            transformer_4D
            )
        # Testset N * C * D * H * W
        self.TestDataLoader = data.DataLoader(
            DatasetFCN_test, 
            batch_size=self.test_batch_size,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            )           
        
    def init_model(self):
        # init
        if self.control=='FCN':
            torch.cuda.set_device(self.args.device_ids[0])
            fcn = _FCN(
                self.in_feature,
                self.hidden_feature,
                self.ts_seq,
                'naive'
            )        
            if len(self.args.device_ids) > 1:
                self.fcn = nn.DataParallel(fcn, device_ids=self.args.device_ids).cuda()    
            else:
                self.fcn = fcn.cuda()
            print(self.args) 
            self.optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
        # init
        if self.control=='iter_FCN':
            torch.cuda.set_device(self.args.device_ids[0])
            fcnIter = _FCN(
                self.in_feature,
                self.hidden_iter_feature,
                self.ts_seq,
                'iter'
            )        
            if len(self.args.device_ids) > 1:
                self.fcn_iter = nn.DataParallel(fcnIter, device_ids=self.args.device_ids).cuda()    
            else:
                self.fcn_iter = fcnIter.cuda()
            print(self.args) 
            self.optimizer = torch.optim.Adam(self.fcn_iter.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)              
              
    def fit(self):
        if self.control=='FCN':
            self.epoch_iterations = 0
            self.init_model()
            # if max_epoch
            for epoch in range(self.max_epoch):
                self.epoch = epoch
                self.train_fcn()
                if self.max_epoch % self.training_inv == 0:
                    self.inference_fcn()
                self.epoch_iterations = self.epoch_iterations + 1 

        if self.control=='iter_FCN':
            self.epoch_iterations = 0
            self.init_model()
            # if max_epoch
            for epoch in range(self.max_epoch):
                self.epoch = epoch
                self.train_fcn_iter()
                if self.max_epoch % self.training_inv == 0:
                    self.inference_fcn_iter()
                self.epoch_iterations = self.epoch_iterations + 1                 
                
    ########Train FCN###########            
    def train_fcn(self):
        self.fcn.train()
        for batch_idx, (data, target) in tqdm.tqdm(
            enumerate(self.TrainDataLoader), 
            total=len(self.TrainDataLoader), 
            desc='Train FCN epoch=%d' % self.epoch, 
            ncols=100, 
            leave=False):
            ## Data
            data = data[:,:,-1,:,:].to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            preds = self.fcn(data, 'naive')
            ## Loss                      
            Loss = F.mse_loss(preds, target)
            #
            Loss.backward()
            self.optimizer.step()            
            self.writer.add_scalars('Loss_{}'.format("FCN"), {
                'fcn_loss': Loss.item()},
                self.epoch_iterations)      
        print ('epoch-{}-last_miniBatch-MSELoss:'.format(self.epoch), Loss.item())
                         
                         
    ########Evaluation FCN###########            
    def inference_fcn(self):
        self.fcn.eval()
        mae_best = 100000 
        batch_preds = []
        batch_gts = []        
        for batch_idx, (data, target) in tqdm.tqdm(
            enumerate(self.TestDataLoader), total = len(self.TestDataLoader),
            desc = 'Test FCN epoch=%d' % self.epoch, ncols = 80,
            leave=False): 
            
            ## Data
            data = data[:,:,-1,:,:].to(device=self.device)
            target = target.to(device=self.device)
            
            # Inference
            with torch.no_grad():
                preds = self.fcn(data, 'naive')
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
        ts_01 = cal_TS_score(preds, labelsVec, 0.1)
        ts_1 = cal_TS_score(preds, labelsVec, 1)
        ts_10 = cal_TS_score(preds, labelsVec, 10)


        info = {"MAE": mae,
                "MPAE": pMae,
                "Ts0.1": ts_01,
                "Ts1:": ts_1,
                "Ts10:": ts_10,
                }
        print("======= FCN Results Show =======".format(self.epoch_iterations + 1))
        print(info)                                                    

    # visual            
        self.writer.add_scalars(
            'Accuracy_{}'.format(self.benchmark_name1), 
            {
            "MAE": mae,
            "MPAE": pMae,
            "Ts0.1": ts_01,
            "Ts1:": ts_1,
            "Ts10:": ts_10}, 
            self.epoch_iterations + 1) 

        if mae < mae_best:
            # return dictionary
            dict_params_fcn = self.fcn.state_dict()
            torch.save(dict_params_fcn, os.path.join(self.models_save_dir, 'fcn-{}.pt'.format(self.epoch_iterations+1))) 
            
            
            
            
    ########Train Iterative FCN###########            
    def train_fcn_iter(self):
        self.fcn_iter.train()
        for batch_idx, (data, target) in tqdm.tqdm(
            enumerate(self.TrainDataLoader), 
            total=len(self.TrainDataLoader), 
            desc='Train Iterative FCN epoch=%d' % self.epoch, 
            ncols=100, 
            leave=False):
            ## Data
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            preds = self.fcn_iter(data,'iter')
            ## Loss                      
            Loss = F.mse_loss(preds, target)
            #
            Loss.backward()
            self.optimizer.step()            
            self.writer.add_scalars('Loss_{}'.format("iterativeFCN"), {
                'fcn_iter_loss': Loss.item()},
                self.epoch_iterations)      
        print ('epoch-{}-last_miniBatch-MSELoss:'.format(self.epoch), Loss.item())
                         
                         
    ########Evaluation Iterative FCN###########            
    def inference_fcn_iter(self):
        self.fcn_iter.eval()
        mae_best = 100000 
        batch_preds = []
        batch_gts = []        
        for batch_idx, (data, target) in tqdm.tqdm(
            enumerate(self.TestDataLoader), total = len(self.TestDataLoader),
            desc = 'Test Iterative FCN epoch=%d' % self.epoch, ncols = 80,
            leave=False): 
            ## Data
            data = data.to(device=self.device)
            target = target.to(device=self.device)
            # Inference
            with torch.no_grad():
                preds = self.fcn_iter(data, 'iter')
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
        ts_01 = cal_TS_score(preds, labelsVec, 0.1)
        ts_1 = cal_TS_score(preds, labelsVec, 1)
        ts_10 = cal_TS_score(preds, labelsVec, 10)


        info = {"MAE": mae,
                "MPAE": pMae,
                "Ts0.1": ts_01,
                "Ts1:": ts_1,
                "Ts10:": ts_10,
                }
        print("======= Iterative FCN Results Show =======".format(self.epoch_iterations + 1))
        print(info)                                                    

    # visual            
        self.writer.add_scalars(
            'Accuracy_{}'.format('iter'+self.benchmark_name2), 
            {
            "MAE": mae,
            "MPAE": pMae,
            "Ts0.1": ts_01,
            "Ts1:": ts_1,
            "Ts10:": ts_10}, 
            self.epoch_iterations + 1) 

        if mae < mae_best:
            # return dictionary
            dict_params_iter_fcn = self.fcn_iter.state_dict()
            torch.save(dict_params_iter_fcn, os.path.join(self.models_save_dir, 'iter_fcn-{}.pt'.format(self.epoch_iterations+1))) 