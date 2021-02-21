import os
import tqdm
import torch
import pickle
import random
import datetime
from apex.parallel import SyncBatchNorm
import numpy as np
import torch.nn.functional as F
from torch.utils import data
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import math

"""
 随机重采样的效果并不好，原因可能是①「欠拟合」 ②还原会原本数据集loss也有问题,所以有可能不是resample问题
 ③ 实验下F.mse()...? schedule_lr?   
"""
class gribDataset(data.Dataset):
    def __init__(self, args, total_pairs, resampleRange, startDate, endDate, dataPath, split, transform, isTrain):
        
        self.cropDataFileRoot = dataPath    
        self.startDatetime = startDate
        self.endDatetime = endDate
        self.total_pairs = total_pairs
        self.resampleRange = resampleRange        
        self.isTrain = isTrain        
        self.transformer = transform 
        self.rainOneHotClass = np.eye(2).astype(np.float32)           
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
            
        find_index_s = [i for i, x in enumerate(totalFilePathList) if x.find(self.startDatetime[-4:]) == 6]
        find_index_e = [i for i, x in enumerate(totalFilePathList) if x.find(self.endDatetime[-4:]) == 6]
        totalGribFileList = totalFilePathList[find_index_s[0]:(find_index_e[0] + 1)]
        totalGribFilesList = [os.path.join(self.cropDataFileRoot, item, crop_path) for item in totalGribFileList for crop_path in os.listdir(os.path.join(self.cropDataFileRoot, item))]
        
        length = len(totalGribFilesList)
        
        if self.isTrain:
            self.cropDataFileNamesList = [os.path.join(self.cropDataFileRoot, item) for item in (list(totalGribFilesList[0:int(0.2 * cross * length)]) + list(totalGribFilesList[int(0.2 * (cross + 1) * length):length]))]
        else:
            self.cropDataFileNamesList = [os.path.join(self.cropDataFileRoot, item) for item in totalGribFilesList[int(0.2 * length * cross):int(0.2 * length * (cross + 1))]]
            
           
    def loadCropData(self):      
        self.cropDataList = []
        for pickleFileName in tqdm.tqdm(
            self.cropDataFileNamesList, 
            total=len(self.cropDataFileNamesList),
            desc="Load {} Data".format(self.isTrain), 
            ncols=80, 
            leave=False):
            
            cropDataDict = self.openPickleFile(pickleFileName)
            
            # get station-wise xarray crop data
            for cropDataArray in cropDataDict.values():   
                micapsValue = cropDataArray.attrs["prep_gt"]   
                rainFallClass = 0       
                if micapsValue >= 0.2:
                    rainFallClass = 1
                
                cropData = (cropDataArray.values, micapsValue, rainFallClass)
                self.cropDataList.append(cropData)
                len_trainSet = len(self.cropDataList)
#         print ('PRE_RAINDATASET_LENGTH:', len_trainSet)        
#         if self.isTrain:    
#             np.random.seed(0)
#             resample_indice = np.random.choice(np.arange(len_trainSet), self.total_pairs) 
#             self.resample_indice_l = [resample_indice for ii in range(self.total_pairs)]
#             print ('---completing TOTAL PAIRS---') 
#             print ('TRAINSET_LENGTH:', self.total_pairs)
        if self.isTrain:
            print ('TRAINSET_LENGTH:', len_trainSet)                                   
        else:      
            print ('TESTSET_LENGTH:', len_trainSet)
           
    def __len__(self):
        
        return len(self.cropDataList)
#         return self.total_pairs
       
    def __getitem__(self, idx):        
#         if self.isTrain:
#             indice = self.resample_indice_l[idx] 
#             oneData = self.cropDataList[random.choice(indice)]       
#         else:
        oneData = self.cropDataList[idx]
            
        features = torch.from_numpy(oneData[0].astype(np.float32))
        micapsValue = np.array([oneData[1]], dtype = np.float32)

        rainFallClass = np.array(oneData[2], dtype = np.int64)
        rainFallMask = self.rainOneHotClass[oneData[2]]

        if self.transformer:
            Features = self.transformer(features)
        return Features, micapsValue, rainFallClass, rainFallMask 


class ordinalGribDataset(data.Dataset):
    def __init__(self, args, total_pairs, startDate, endDate, dataPath, split, scope, resampleRange, transform):
        
        self.cropDataFileRoot = dataPath
        self.startDatetime = startDate
        self.endDatetime = endDate 
        self.total_pairs = total_pairs
        self.range = scope
        self.resampleRange = resampleRange
        self.resample_indice_l = []
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

    
    def trainTestFileSplit(self,cross = 0):
        
        
        totalGribFileList = []
        
        totalFilePathList = os.listdir(self.cropDataFileRoot)
        totalFilePathList.sort()
    
        find_index_s = [i for i, x in enumerate(totalFilePathList) if x.find(self.startDatetime[-4:]) == 6]
        find_index_e = [i for i, x in enumerate(totalFilePathList) if x.find(self.endDatetime[-4:]) == 6]       
        totalGribFileList = totalFilePathList[find_index_s[0]:(find_index_e[0] + 1)]      
        totalGribFilesList = [os.path.join(self.cropDataFileRoot, item, crop_path) for item in totalGribFileList for crop_path in os.listdir(os.path.join(self.cropDataFileRoot, item))]
                      
        length = len(totalGribFilesList)
        
        # 80%
        self.cropDataFileNamesList = [os.path.join(self.cropDataFileRoot, item) for item in
                                      (list(totalGribFilesList[0:int(0.2 * cross * length)]) + list(totalGribFilesList[int(0.2 * (cross + 1) * length):length]))]

    def loadCropData(self):
        self.cropDataList = []     
        for pickleFileName in tqdm.tqdm(
            self.cropDataFileNamesList, 
            total=len(self.cropDataFileNamesList),
            desc="Load {} Data".format("rainFallOR"), 
            ncols=80, 
            leave=False):
            cropDataDict = self.openPickleFile(pickleFileName)         
            for cropDataArray in cropDataDict.values():
                micapsValue = cropDataArray.attrs["prep_gt"]    
#                 if micapsValue < 0.5:
                if micapsValue < 0.1:
                    continue              
                ordinalLabels = 1.0 * (self.range < micapsValue)
                cropData = (cropDataArray.values, micapsValue, ordinalLabels)
                self.cropDataList.append(cropData)
                
            len_trainSet = len(self.cropDataList)
#         print ('PRE_ORDDATASET_LENGTH:', len_trainSet) 
#         np.random.seed(0)
#         resample_indice = np.random.choice(np.arange(len_trainSet), self.total_pairs)
#         self.resample_indice_l = [resample_indice for ii in range(self.total_pairs)]
#         print ('---completing TOTAL PAIRS---')
#         print ('ORDINAL_DATASET_LENGTH:', self.total_pairs)
        print ('ORDINAL_DATASET_LENGTH:', len_trainSet)

    def __len__(self):
        return len(self.cropDataList)
#         return self.total_pairs
    
    def __getitem__(self, idx):   
#         indice = self.resample_indice_l[idx]
#         oneData = self.cropDataList[random.choice(indice)]
        oneData = self.cropDataList[idx]
        features = torch.from_numpy(oneData[0].astype(np.float32))
        micapsValue = np.array([oneData[1]], dtype=np.float32)        
        ordinalLabels = oneData[2].astype(np.float32)

        if self.transformer:
            Features = self.transformer(features)
        return Features, micapsValue, ordinalLabels
   
