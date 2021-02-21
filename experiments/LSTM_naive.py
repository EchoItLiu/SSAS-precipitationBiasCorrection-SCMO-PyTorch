'''
'''
import os
import os.path as osp
from torch.utils import data
import torch
import torch.nn as nn
from models.CompModels import _LSTM, _ConvLSTM
import numpy as np
from torchvision import transforms
import pickle
import tqdm
import random
import sys
import torch.distributed as dist
from utils.util_main import featuresNorm
from utils.util_benchmarks import cal_PMAE, cal_TS_score, cal_MAE, filePathGenerate, dataGenerate, ycluo2_dataLoader, deepModels_dataLoader, data_interplote_nor_gt, splitBatch, floorHalf
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import math

class LSTM(object):
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
                 in_feature_size_lstm,
                 hidden_feature_size_lstm,
                 in_feature_size_convLstm,
                 hidden_feature_size_convLstm,
                 ts_seq,
                 #
                 train_batch_size_lstm_singleGPU,
                 test_batch_size_lstm_singleGPU,
                 train_batch_size_convlstm,
                 test_batch_size_convlstm,
                 #
                 training_inv,
                 learning_rate_lstm,
                 learning_rate_convLstm,
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
#         self.dataset_path = osp.join(args.basePath, args.dataset_name, dataset_type)
#         self.dataset_path = '/mnt/pami23/yqliu/cropdata_ts'
        self.dataset_path = '/mnt/pami23/yqliu/ycluo/cropdata_ts'
    
        
        self.dataset_util = dataset_util
        self.startDate = startDate
        self.endDate = endDate
        self.ECSize = EC_size
        self.split_ratio = split_ratio 
        self.crop_scale = crop_scale
        #
        self.in_feature_size_lstm = in_feature_size_lstm
        self.hidden_feature_size_lstm = hidden_feature_size_lstm
        self.in_feature_size_convLstm = in_feature_size_convLstm
        self.hidden_feature_size_convLstm = hidden_feature_size_convLstm
        self.ts_seq = ts_seq        
        self.train_batch_size_lstm_singleGPU = train_batch_size_lstm_singleGPU
        self.test_batch_size_lstm_singleGPU = test_batch_size_lstm_singleGPU
        self.train_batch_size_convlstm = train_batch_size_convlstm
        self.test_batch_size_convlstm = test_batch_size_convlstm        
        self.training_inv = training_inv
        self.learning_rate_lstm = learning_rate_lstm
        self.learning_rate_convLstm = learning_rate_convLstm        
        self.weight_decay = weight_decay
        
        self.max_epoch = 50
        
        self.num_workers = args.num_workers 
        #
        self.EC_center = floorHalf(self.ECSize)
        #
        self.writer = SummaryWriter(osp.join(self.args.save_root, 'tensorboard'))
#         self.models_save_dir = osp.join(self.args.save_root, 'save_models')
#         os.makedirs(self.models_save_dir, exist_ok=True)        
        # LSTM
        self.benchmark_name1 = benchmark_name1
        # convLSTM
        self.benchmark_name2 = benchmark_name2        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #############
#         self.control = 'lstm'
        self.control = 'convlstm'
        #############
    
        
    def __repr__(self):
        return self.__class__.__name__

    def init_module(self):
        if self.control=='lstm':
           # crop train/test
            totalGribFileList = filePathGenerate(self.dataset_path, self.startDate, self.endDate)
           # Dataset split
            trainCropFileList, testCropFileList = dataGenerate(totalGribFileList, self.split_ratio)
#             print ('trainCropFileList:', trainCropFileList)
#             print ('testCropFileList:', testCropFileList)
            ## LSTM ND * C 
            self.featuresArr_train, self.labelsVec_train = data_interplote_nor_gt(trainCropFileList, self.EC_center, self.crop_scale, self.benchmark_name1)
            self.featuresArr_test, self.labelsVec_test = data_interplote_nor_gt(testCropFileList, self.EC_center, self.crop_scale, self.benchmark_name1)
        ##
        if self.control=='convlstm': 
            ## ConvLstm B*C*D*H*W
            featuresMean_st, featuresStd_st = featuresNorm(self.args, self.dataset_name, self.dataset_util, 'ycluo37')
            ##△ ycluo
#             featuresMean_st, featuresStd_st = featuresNorm(self.args, self.dataset_name, self.dataset_util, 'ycluo2')
            # 4DTransformer
            transformer_4D = transforms.Compose([
            transforms.Normalize(mean=featuresMean_st, std=featuresStd_st)
        ])
            ##△ ycluo
#             transformer_luo = transforms.Compose([
#             transforms.Normalize(mean=featuresMean_st, std=featuresStd_st)
#             ])  
           # dataLoader
            totalGribFileList = filePathGenerate(self.dataset_path, self.startDate, self.endDate)
           # Dataset split
            trainCropFileList, testCropFileList = dataGenerate(totalGribFileList, self.split_ratio)             # Trainset 
            DatasetConvLSTM_train = deepModels_dataLoader(self.args,
                trainCropFileList,
                transformer_4D,
                'train'
                )
              ##△ ycluo
#             DatasetConvLSTM_train = ycluo2_dataLoader(
#                 trainCropFileList,
#                 transformer_luo,
#                 3                                     
#                 )            
            self.lstmTrainDataLoader = data.DataLoader(
                DatasetConvLSTM_train, 
                batch_size=self.train_batch_size_convlstm,
                drop_last=True,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
                )  
        
            # Testset
            DatasetConvLSTM_test = deepModels_dataLoader(
                self.args,
                testCropFileList,
                transformer_4D,
                'test'
                )
            ##△ ycluo            
#             DatasetConvLSTM_test = ycluo2_dataLoader(
#                 testCropFileList,
#                 transformer_luo,
#                 3
#                 )
            self.lstmTestDataLoader = data.DataLoader(
                DatasetConvLSTM_test, 
                batch_size=self.test_batch_size_convlstm,
                drop_last=True,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
                )           
        
    def init_model(self):
        if self.control=='lstm':
            # LSTM
            torch.cuda.set_device(self.args.device_ids[0])
            lstm = _LSTM(
                self.in_feature_size_lstm,
                self.hidden_feature_size_lstm)        
#             if len(self.args.device_ids) > 1:
#                 self.lstm = nn.DataParallel(lstm, device_ids=self.args.device_ids).cuda()    
#             else:
            self.lstm = lstm.cuda()
            print(self.args) 
            self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate_lstm, weight_decay=self.weight_decay)
                                                           
        # ConvLSTM
        if self.control=='convlstm':
            torch.cuda.set_device(self.args.device_ids[0])                                              
            conv_lstm = _ConvLSTM(
                6,
                37,
                self.hidden_feature_size_lstm,
                19,
                self.device     
                )
            if len(self.args.device_ids) > 1:
                self.conv_lstm = nn.DataParallel(conv_lstm, device_ids=self.args.device_ids).cuda()    
            else:
                self.conv_lstm = conv_lstm.cuda()
            print(self.args) 
            self.optimizer_convlstm = torch.optim.Adam(self.conv_lstm.parameters(), lr=self.learning_rate_convLstm, weight_decay=self.weight_decay)
                                                           
        
    def fit(self):
        self.init_model()
        #                                                   
        if self.control=='lstm':
            self.epoch_iterations = 0
            for epoch in range(self.max_epoch):
                self.epoch = epoch
                self.train_lstm()
                if self.max_epoch % self.training_inv == 0:
                    self.inference_lstm()
                self.epoch_iterations = self.epoch_iterations + 1
                                                           
        if self.control=='convlstm':
            self.epoch_iterations = 0
            for epoch in range(self.max_epoch):
                self.epoch = epoch
                self.train_convlstm()
                if self.max_epoch % self.training_inv == 0:
                    self.inference_convlstm()
                self.epoch_iterations = self.epoch_iterations + 1                                                           
        
    ########Train LSTM###########
    def train_lstm(self):                                                       
        self.lstm.train()
        '''
        单GPU dataLoader
        '''
        # split batch to be sampler
        train_batch_list = splitBatch(self.featuresArr_train, self.labelsVec_train, self.train_batch_size_lstm_singleGPU, self.ts_seq, 'Train', self.epoch)
        #                                                   
        for batch_idx, (train_data, train_target) in tqdm.tqdm(
            enumerate(train_batch_list), 
            total=len(train_batch_list), 
            desc='Train LSTM epoch=%d' % self.epoch, 
            ncols=100, 
            leave=False):            
            ## Forward
            self.optimizer.zero_grad()
            
            preds = self.lstm(train_data)
            
            ## Loss                      
            Loss = F.mse_loss(preds, train_target)
            #
            Loss.backward()
            self.optimizer.step()
            
            self.writer.add_scalars('Loss_{}'.format("LSTM"), {
                'lstm_loss': Loss.item()},
                self.epoch_iterations)      
        print ('epoch-{}-last_miniBatch-MSELoss:'.format(self.epoch), Loss.item())                                                           
                                                           
    ########Evaluation LSTM###########            
    def inference_lstm(self):
        self.lstm.eval()
        test_batch_list = splitBatch(self.featuresArr_test, self.labelsVec_test, self.test_batch_size_lstm_singleGPU, self.ts_seq, 'Test', self.epoch)                                 
        mae_best = 100000
        batch_preds = []
        batch_gts = []
        
        for batch_idx, (test_data, test_target) in tqdm.tqdm(
            enumerate(test_batch_list), 
            total=len(test_batch_list), 
            desc='Test LSTM epoch=%d' % self.epoch, 
            ncols=100, 
            leave=False): 
                
            with torch.no_grad():
                preds = self.lstm(test_data)                                                 
                Loss = F.mse_loss(preds, test_target).item()
                #
                batch_preds.append(preds.cpu().numpy().squeeze())
                batch_gts.append(test_target.cpu().numpy().squeeze())
                                                           
        
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
        print("======= LSTM Results Show =======".format(self.epoch_iterations + 1))
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

#         if mae < mae_best:
#             # return dictionary
#             dict_params_lstm = self.lstm.state_dict()
#             torch.save(dict_params_lstm, os.path.join(self.models_save_dir, 'lstm-{}.pt'.format(self.epoch_iterations+1))) 

            # return dictionary
        dict_params_lstm = self.lstm.state_dict()
#         torch.save(dict_params_lstm, os.path.join(self.models_save_dir, 'lstm-{}.pt')) 
#                                                            
                                                           
                                                                     
    ########Train ConvLSTM###########                                                         
    def train_convlstm(self):
        self.conv_lstm.train()
        for batch_idx, (data, target) in tqdm.tqdm(
            enumerate(self.lstmTrainDataLoader), 
            total=len(self.lstmTrainDataLoader), 
            desc='Train Conv LSTM epoch=%d' % self.epoch, 
            ncols=100, 
            leave=False):
            ## Data
            data = data.to(self.device)
            target = target.to(self.device)
            
            ## Forward
            self.optimizer_convlstm.zero_grad()
            
            preds = self.conv_lstm(data)
            ## Loss                      
            Loss = F.mse_loss(preds, target)
            #
            Loss.backward()
            self.optimizer_convlstm.step() 
            #                                               
#             self.writer.add_scalars('Loss_{}'.format("ConvLSTM"), {
#                 'convLSTM_loss': Loss.item()},
#                 self.epoch_iterations)      
        print ('epoch-{}-last_miniBatch-MSELoss:'.format(self.epoch), Loss.item())
#         print ('epoch-{}-last_miniBatch-MSELoss:'.format(self.epoch), 7.3455159)
                         
                         
    ########Evaluation ConvLSTM###########            
    def inference_convlstm(self):
        self.conv_lstm.eval()
        mae_best = 100000 
        batch_preds = []
        batch_gts = []                                                   
        for batch_idx, (data, target) in tqdm.tqdm(
            enumerate(self.lstmTestDataLoader), total = len(self.lstmTestDataLoader),
            desc = 'Test Conv LSTM epoch=%d' % self.epoch, ncols = 80,
            leave=False): 
            ## Data
            data = data.to(device=self.device)
            ##
            target = target.numpy()
            ##
#             print ('test_target:', target)
            
            # Inference
            with torch.no_grad():
                preds_test = self.conv_lstm(data)
                batch_preds.append(preds_test.squeeze().cpu().numpy())
                batch_gts.append(target)                
                
                
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
            
        print("======= Convolutional LSTM Results Show =======".format(self.epoch_iterations + 1))
        print(info) 
        
        f1 = open(os.path.join('/mnt/pami23/yqliu/out_convlstm', 'out4.txt'), 'a')
        
        # 必须是string类型才能保存，读取用eval()
        f1.write(str(info))
#         f2.write(str(info_best))
        ##
        f1.close()
#         f2.close()
        
        
    # visual            
#         self.writer.add_scalars(
#             'Accuracy_{}'.format(self.benchmark_name2), 
#             {
#             "MAE": mae,
#             "MPAE": pMae,
#             "Ts0.1": ts_01,
#             "Ts1:": ts_1,
#             "Ts10:": ts_10}, 
#             self.epoch_iterations+1) 

#         if mae < mae_best:
#             # return dictionary
#             dict_params_conv_lstm  = self.conv_lstm.state_dict()
#             torch.save(dict_params_conv_lstm, os.path.join(self.models_save_dir, 'convlstm-{}.pt'.format(self.epoch_iterations+1)))
#         dict_params_conv_lstm  = self.conv_lstm.state_dict()
#         torch.save(dict_params_conv_lstm, os.path.join(self.models_save_dir, 'convlstm-{}.pt')) 