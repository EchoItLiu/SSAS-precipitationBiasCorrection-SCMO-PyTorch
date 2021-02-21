import os
import tqdm
import torch
import pickle
import random
import datetime
from utils.util_main import ceilHalf, floorHalf
from apex.parallel import SyncBatchNorm
import numpy as np
import torch.nn.functional as F
from torch.utils import data
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import math

class modalGribDataset(data.Dataset):
    def __init__(self, args, total_pairs, resampleRange, startDate, endDate, dataPath, split, transform, isTrain):
        
        self.cropDataFileRoot = dataPath    
        self.startDatetime = startDate
        self.endDatetime = endDate
        self.total_pairs = total_pairs
        self.resampleRange = resampleRange
        self.isTrain = isTrain        
        self.rainOneHotClass = np.eye(2).astype(np.float32)  
        self.transformer = transform
        ##
        self.trainTestFileSplit(split) 
        self.loadCropData()
               
    def openPickleFile(self, pickleFileName):
        
        if not os.path.exists(pickleFileName):
            raise NameError("No %s file exists." % (pickleFileName))

        with open(pickleFileName, 'rb') as f:
            oneCropData = pickle.load(f)

        return oneCropData
        
            
    def trainTestFileSplit(self, cross):
        totalGribFileList = []      
        totalFilePathList = os.listdir(self.cropDataFileRoot)
        totalFilePathList.sort()
            
        find_index_s = [i for i, x in enumerate(totalFilePathList) if x.find(self.startDatetime[-4:]) == 14]
        find_index_e = [i for i, x in enumerate(totalFilePathList) if x.find(self.endDatetime[-4:]) == 14] 
        totalGribFileList = totalFilePathList[find_index_s[0]:(find_index_e[0] + 1)]
        totalGribFilesList = [os.path.join(self.cropDataFileRoot, item, crop_path) for item in totalGribFileList for crop_path in os.listdir(os.path.join(self.cropDataFileRoot, item))]
        
        length = len(totalGribFilesList)
        
        if self.isTrain:
            self.cropDataFileNamesList = [os.path.join(self.cropDataFileRoot, item) for item in (list(totalGribFilesList[0:int(0.2 * cross * length)]) + list(totalGribFilesList[int(0.2 * (cross + 1) * length):length]))]
        else:
            self.cropDataFileNamesList = [os.path.join(self.cropDataFileRoot, item) for item in totalGribFilesList[int(0.2 * length * cross):int(0.2 * length * (cross + 1))]]
            
           
    def loadCropData(self):      
        self.modalDataList = []
        for pickleFileName in tqdm.tqdm(
            self.cropDataFileNamesList, 
            total=len(self.cropDataFileNamesList),
            desc="Load {} Data".format(self.isTrain), 
            ncols=80, 
            leave=False):
            
            cropDataDict = self.openPickleFile(pickleFileName)
            
            # get station-wise xarray crop data
            for cropDataArray in cropDataDict.values():
                # C * D * h * w
                EC_features = cropDataArray.values
                seq_micapsValue = cropDataArray.attrs["seq_prep_gt"] 
                seq_tempers = cropDataArray.attrs['seq_tem_gt']
                seq_press = cropDataArray.attrs['seq_press_gt']
                seq_wind = cropDataArray.attrs['seq_wind_gt']
                seq_dew = cropDataArray.attrs['seq_dew_gt']
                ## 对每个元素操作
                seq_rainFallClass = [1 if mi_v>=0.2 else 0 for i, mi_v in enumerate(seq_micapsValue)]
                
                # 4Rain Binary
                if self.isTrain==True:
                    seq_micapsValue = seq_micapsValue[-1]
                    seq_rainFallClass = seq_rainFallClass[-1]
                    EC_features = EC_features[:,-1,:,:]  
                    cropData = [EC_features, seq_micapsValue, seq_rainFallClass]
                # 4Rain Binary
                else:    
                    cropData = [EC_features, seq_micapsValue, seq_rainFallClass, seq_tempers, seq_press, seq_wind, seq_dew]
                                              
                self.modalDataList.append(cropData)
                len_Set = len(self.modalDataList)
        # h/w
        if self.isTrain==False:
            
            self.seq_length = self.modalDataList[0][0].shape[1] 
            self.channel_nums = self.modalDataList[0][0].shape[0]  
            self.currScale = self.modalDataList[0][0].shape[3] 

        
        # TRAIN:  [(C * h * w, 1,[],[],[(.,.)]..), .., ..]
        # TEST: [(C * D * h * w),[],[],[(.,.)]..], .., ..]
        if self.isTrain:
            print ('TRAINSET_LENGTH:', len_Set)                                   
        else:      
            print ('TESTSET_LENGTH:', len_Set)

    def __len__(self):
        return len(self.modalDataList)
       
    def __getitem__(self, idx):        
        oneData = self.modalDataList[idx]
        if self.isTrain==False:
            features = torch.from_numpy(oneData[0].astype(np.float32))
            seq_micapsValue = np.array(oneData[1], dtype = np.float32)       
            seq_rainFallClass = np.array(oneData[2], dtype = np.int64)
            seq_tempers = np.array(oneData[3], dtype = np.float32)
            seq_press = np.array(oneData[4], dtype = np.float32)
            # [[...]]
            seq_wind = np.array(oneData[5], dtype = np.float32)
            seq_dew = np.array(oneData[6], dtype = np.float32)
            features_C = features.view(self.channel_nums, self.seq_length * self.currScale, self.currScale)
            nor_features = self.transformer(features_C)
            Features = nor_features.view(self.channel_nums, self.seq_length, self.currScale, self.currScale)  
            return Features, seq_micapsValue, seq_rainFallClass, seq_tempers, seq_press, seq_wind, seq_dew 
    
        else:
            features = torch.from_numpy(oneData[0].astype(np.float32))
            micapsValue = np.array(oneData[1], dtype = np.float32)       
            rainFallClass = np.array(oneData[2], dtype = np.int64)
            Features = self.transformer(features)
            return Features, micapsValue, rainFallClass
            
           
                   
class modalOrdinalDataset(data.Dataset):
    def __init__(self, args, total_pairs, startDate, endDate, dataPath, split, scope, resampleRange, transform):
        
        self.cropDataFileRoot = dataPath
        self.startDatetime = startDate
        self.endDatetime = endDate 
        self.total_pairs = total_pairs
        self.range = scope
        self.resampleRange = resampleRange
        self.resample_indice_l = []
        self.transformer = transform
        #
        self.trainTestFileSplit(split)
        self.loadCropData()
        
    def openPickleFile(self, pickleFileName):
        if not os.path.exists(pickleFileName):
            raise NameError("No %s file exists." % (pickleFileName))

        with open(pickleFileName, 'rb') as f:
            oneCropData = pickle.load(f)
        return oneCropData
   
    def trainTestFileSplit(self,cross = 0):
          
        totalGribFileList = []
        
        totalFilePathList = os.listdir(self.cropDataFileRoot)
        totalFilePathList.sort()
        ##
        find_index_s = [i for i, x in enumerate(totalFilePathList) if x.find(self.startDatetime[-4:]) == 14]
        find_index_e = [i for i, x in enumerate(totalFilePathList) if x.find(self.endDatetime[-4:]) == 14]       
        totalGribFileList = totalFilePathList[find_index_s[0]:(find_index_e[0] + 1)]      
        totalGribFilesList = [os.path.join(self.cropDataFileRoot, item, crop_path) for item in totalGribFileList for crop_path in os.listdir(os.path.join(self.cropDataFileRoot, item))]    
        length = len(totalGribFilesList)
        # 80%
        self.cropDataFileNamesList = [os.path.join(self.cropDataFileRoot, item) for item in
                                      (list(totalGribFilesList[0:int(0.2 * cross * length)]) + list(totalGribFilesList[int(0.2 * (cross + 1) * length):length]))]

    def loadCropData(self):
        self.modalOrdList = []     
        for pickleFileName in tqdm.tqdm(
            self.cropDataFileNamesList, 
            total=len(self.cropDataFileNamesList),
            desc="Load {} Data".format("rainFallOR"), 
            ncols=80, 
            leave=False):
            cropDataDict = self.openPickleFile(pickleFileName)         
            for cropDataArray in cropDataDict.values():
                
                seq_micapsValue = cropDataArray.attrs["seq_prep_gt"]
                '''
                OR加了带0的样本？？？
                '''
                seq_micapsValue = [0 if mi_v<=0.1 else mi_v for i, mi_v in enumerate(seq_micapsValue)]                
                seq_tempers = cropDataArray.attrs['seq_tem_gt']
                seq_press = cropDataArray.attrs['seq_press_gt']
                seq_wind = cropDataArray.attrs['seq_wind_gt']
                seq_dew = cropDataArray.attrs['seq_dew_gt']
                
                '''
                这里样本包括带0的
                '''
                seq_ordinalLabels = [1.0 * (self.range < mi_v) for mi_v in seq_micapsValue]
                           
                cropData = [cropDataArray.values, seq_micapsValue, seq_ordinalLabels, seq_tempers, seq_press, seq_wind, seq_dew]                
                self.modalOrdList.append(cropData)
        
        self.seq_length = self.modalOrdList[0][0].shape[1] 
        self.channel_nums = self.modalOrdList[0][0].shape[0]  
        self.currScale = self.modalOrdList[0][0].shape[3] 
               
        len_trainSet = len(self.modalOrdList)
            
        # ORD:  [(C * D * h * w, [],[],[1-nd,.].., .., ..]
        print ('ORDINAL_DATASET_LENGTH:', len_trainSet)
        
    def __len__(self):
        return len(self.modalOrdList)
    
    def __getitem__(self, idx):       
        oneData = self.modalOrdList[idx]
        # Tensor: C * D * h_s * w_s
#         print (oneData, oneData)
        features = torch.from_numpy(oneData[0].astype(np.float32))
        seq_micapsValue = np.array(oneData[1], dtype = np.float32)       
        seq_ordinalLabels = np.array(oneData[2], dtype = np.float32)
        seq_tempers = np.array(oneData[3], dtype = np.float32)
        seq_press = np.array(oneData[4], dtype = np.float32)
        # [[...]]
        seq_wind = np.array(oneData[5], dtype = np.float32)
        seq_dew = np.array(oneData[6], dtype = np.float32)
        ##
        features_C = features.view(self.channel_nums, self.seq_length * self.currScale, self.currScale)
        nor_features = self.transformer(features_C)
        Features = nor_features.view(self.channel_nums, self.seq_length, self.currScale, self.currScale)
                                   
        return Features, seq_micapsValue, seq_ordinalLabels, seq_tempers, seq_press, seq_wind, seq_dew
            
    
    
      