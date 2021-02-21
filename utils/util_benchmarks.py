import os
import torch
import pickle
import tqdm
import random
import sys
import numpy as np
from torch.utils import data
from collections import Counter
import sys
import json
import torch.distributed as dist
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer


def filePathGenerate(datasetPath, startDate, endDate):
    # 截取日期
#     totalGribDirList = os.listdir(datasetPath)
#     totalGribDirList.sort()
#     find_index_s = [i for i, x in enumerate(totalGribDirList) if x.find(startDate[-4:]) == 14]
#     find_index_e = [i for i, x in enumerate(totalGribDirList) if x.find(endDate[-4:]) == 14] 
#     totalGribDirList = totalGribDirList[find_index_s[0]:(find_index_e[0] + 1)]
#     totalGribDirList = ['0-modals-seq201608','1-modals-seq201609','2-modals-seq201610','8-modals-seq201708','9-modals-seq201709']
    totalGribDirList = ['0-seq201608','1-seq201609','2-seq201610']

    ##
    totalGribFilesList = [os.path.join(datasetPath, item, crop_path) for item in totalGribDirList for crop_path in os.listdir(os.path.join(datasetPath, item))]
    ##
    return totalGribFilesList
    #△ ycluo
#     return totalGribDirList


def dataGenerate(totalGribFileList, splitRatio):
    random.seed(0)
    random.shuffle(totalGribFileList)
    cropFileList = totalGribFileList[:int(splitRatio * len(totalGribFileList))]
    testCropFileList = totalGribFileList[-int((1-splitRatio) * len(totalGribFileList)):]
    ##△ ycluo
#     cropDataFileRoot = '/mnt/pami14/yqliu/dataset_meteo/cropGrib'
#     length = len(totalGribFileList) 
#     cropFileList = [os.path.join(cropDataFileRoot, item) for item in
#                                   (list(totalGribFileList[0:int(0.2 * 4 * length)]) + list(totalGribFileList[int(0.2 * (4 + 1) * length):length]))]
    
#     testCropFileList = [os.path.join(cropDataFileRoot, item) for item in totalGribFileList[int(0.2 * length * 4):int(0.2 * length * (4 + 1))]]
    return cropFileList, testCropFileList

# def dataGenerate_ts(totalGribFileList, splitRatio):
# #     random.seed(0)
# #     random.shuffle(totalGribFileList)
# #     cropFileList = totalGribFileList[:int(splitRatio * len(totalGribFileList))]
# #     testCropFileList = totalGribFileList[-int((1-splitRatio) * len(totalGribFileList)):]
#     ##△ ycluo
#     cropDataFileRoot = '/mnt/pami14/yqliu/dataset_meteo/cropGrib'
#     length = len(totalGribFileList) 
#     cropFileList = [os.path.join(cropDataFileRoot, item) for item in
#                                   (list(totalGribFileList[0:int(0.2 * 4 * length)]) + list(totalGribFileList[int(0.2 * (4 + 1) * length):length]))]
    
#     testCropFileList = [os.path.join(cropDataFileRoot, item) for item in totalGribFileList[int(0.2 * length * 4):int(0.2 * length * (4 + 1))]]

#     return cropFileList, testCropFileList

# ConvLSTM/FPN/FCN
class deepModels_dataLoader(data.Dataset):
    def __init__(self, args, cropFileList, transform, flag):
        self.cropDataFileNamesList = cropFileList
        self.transformer = transform
        self.flag = flag
        #
        self.loadCropData()
        
    def openPickleFile(self, pickleFileName):
        if not os.path.exists(pickleFileName):
            raise NameError("No %s file exists." % (pickleFileName))
        with open(pickleFileName, 'rb') as f:
            oneCropData = pickle.load(f)
        return oneCropData

    def loadCropData(self):
        self.cropDataList = []     
        for pickleFileName in tqdm.tqdm(
            self.cropDataFileNamesList, 
            total=len(self.cropDataFileNamesList),
            desc="Load {} Data".format("deepBaselines"), 
            ncols=80, 
            leave=False):
            #
            cropDataDict = self.openPickleFile(pickleFileName)         
            for cropDataArray in cropDataDict.values():
#                 micapsValue = cropDataArray.attrs["seq_prep_gt"][-1]
                micapsValue = cropDataArray.attrs["prep_gt"] 

                if self.flag == 'train' and micapsValue < 0.1:
                    continue
                # N * C
                cropData = (np.concatenate((cropDataArray.values[0:35], cropDataArray.values[55:]),0), micapsValue)
                test_feature_size = np.concatenate((cropDataArray.values[0:35], cropDataArray.values[55:]),0).shape
#                 print ('test_feature_size:', test_feature_size)
                self.cropDataList.append(cropData)
                
        self.seq_length = self.cropDataList[0][0].shape[1] 
        self.channel_nums = self.cropDataList[0][0].shape[0]  
        self.currScale = self.cropDataList[0][0].shape[3]
        ##
        print ('cropdatalistLength-{}'.format(self.flag), len(self.cropDataList))
            
    def __len__(self):
        return len(self.cropDataList)
    
    def __getitem__(self, idx):   
        oneData = self.cropDataList[idx]
        features = torch.from_numpy(oneData[0].astype(np.float32))
        features_C = features.view(self.channel_nums, self.seq_length * self.currScale, self.currScale)
        nor_features = self.transformer(features_C)
        Features = nor_features.view(self.channel_nums, self.seq_length, self.currScale, self.currScale)          
        #
        micapsValue = np.array([oneData[1]], dtype=np.float32)        
        return Features, micapsValue
    
'''
不同于benchmark的监督是ERA5的每个格点都有GT，这儿只有站点，所以必须还是得做中心化插值
'''        
def data_interplote_nor_gt(cropFileList, EC_center, crop_scale, flag):
    featuresList = []
    labelsList = []
    for fileName in tqdm.tqdm(cropFileList[:], 
                              total=len(cropFileList),
                              desc="FileName", 
                              ncols=80, 
                              leave=False):
        
        with open(fileName, 'rb') as f:
            oneCropDataDict = pickle.load(f)
        for cropDataArray in tqdm.tqdm(oneCropDataDict.values(), total=len(oneCropDataDict),
                                       desc="One File Data", ncols=80, leave=False):
            
            if flag=='ECOrigin':
                # 1
                meanCropFeature = cropECPrep_mean_4Sample(cropDataArray, crop_scale, EC_center)
                
            elif flag=='LSTM':
                # C*D
                meanCropFeature = cropcenterScale_mean_4Sample_lstm(cropDataArray.values, crop_scale, EC_center)
         
            else:
                meanCropFeature = cropcenterScale_mean_4Sample(cropDataArray.values, crop_scale, EC_center)       
            #
            featuresList.append(meanCropFeature)
            micapsValue = cropDataArray.attrs["micapValues"][-1]                
            labelsList.append(micapsValue)
    # N * C / ND * C / N 
    featuresArr = np.array(featuresList).astype(np.float32)    
    # N * C * D → ND * C  
    if flag=='LSTM':
        arr_num = featuresArr.shape[0]        
        arr_fSize = featuresArr.shape[1]
        arr_seq = featuresArr.shape[2]        
        featuresArr = np.transpose(featuresArr,(0,2,1)).reshape(-1, arr_fSize)
        
    if flag[0:8]=='Bayesian':
        interval_rain_split = int(flag[8:])
        cls_label_ranges = np.arange(0,90,interval_rain_split)
        cls_labels_ranges = [(i,st, st+interval_rain_split) for i,st in enumerate(cls_label_ranges)]
        labelList_update = []
        orgin_len_labels = len(labelsList)
        for label in labelsList:
            for idx in range(len(cls_labels_ranges)):
                if label>cls_labels_ranges[idx][1] and label<cls_labels_ranges[idx][2]:
                    update_label = cls_labels_ranges[idx][0]
                    labelList_update.append(update_label)
                    break                   
                if label==cls_labels_ranges[idx][2] and label==cls_labels_ranges[idx+1][1]:
                    update_label = cls_labels_ranges[idx][0]
                    labelList_update.append(update_label)
                    break      
                if label==0:
                    update_label = label
                    labelList_update.append(update_label)
                    break                          
                if label>=90:
                    labelList_update.append(update_label)
                    update_label = cls_labels_ranges[-1][0]
                    break
        # n
        labelsList = labelList_update
        print (len(labelsList)==orgin_len_labels)
    #    
    if flag!='ECOrigin':
        scaler = StandardScaler()
        # Normalization
        scaler.fit(featuresArr)
        featuresArr = scaler.transform(featuresArr) 
    # n
    labelsVec = np.array(labelsList).astype(np.float32)
    return featuresArr, labelsVec


def iterData_ts_gt_nor(cropFileList, cropScale, EC_center, in_channel, seq):
    featuresList = []
    labelsList = []
#     featuresNorList = []
    for fileName in tqdm.tqdm(cropFileList[:], 
                              total=len(cropFileList),
                              desc="FileName", 
                              ncols=80, 
                              leave=False):
        
        with open(fileName, 'rb') as f:
            oneCropDataDict = pickle.load(f)
        for cropDataArray in tqdm.tqdm(oneCropDataDict.values(), total=len(oneCropDataDict),
                                       desc="One File Data", ncols=80, leave=False):
                  
            # n*C*D | n*D
            meanCropFeature = cropcenterScale_mean_4Sample_lstm(cropDataArray.values, cropScale, EC_center)   
            featuresList.append(meanCropFeature)
            micapsValue_seq = cropDataArray.attrs["seq_prep_gt"]
            EC_ts = len(micapsValue_seq)
            labelsList.append(micapsValue_seq)
    # nD * C 
    featuresArr = np.stack(featuresList).transpose((0,2,1)).reshape(-1, in_channel)
#     print ('featuresArr_size:', featuresArr.shape)
#     print ('featuresArr:', featuresArr)
    # imputation along column
#     my_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
#     featuresArr = my_imputer.fit_transform(featuresArr)
#     print ('imputer_featuresArr_size:', featuresArr.shape)
#     print ('imputer_featuresArr:', featuresArr) 
    # Nor.
    scaler = StandardScaler()
    scaler.fit(featuresArr)
    featuresArr_nor = scaler.transform(featuresArr)
    # D * n * C
    featuresArr_nor = featuresArr_nor.reshape(-1, seq, in_channel).transpose((1,0,2)).astype(np.float32)
    # D*n
    labelsArr = np.array(labelsList).T
    return featuresArr_nor, labelsArr    
    

def cropcenterScale_mean_4Sample_lstm(station_sample, cropScale, EC_center):
    # crop C * D * H * W →  C * D * Hs * Ws
    curr_loc_l = EC_center - ceilHalf(cropScale)
    curr_loc_r = EC_center + ceilHalf(cropScale)
    curr_loc_t = EC_center + ceilHalf(cropScale)
    curr_loc_b = EC_center - ceilHalf(cropScale)

    crop_sample = station_sample[:, :, curr_loc_l:(curr_loc_r+1), curr_loc_b:(curr_loc_t+1)]
    # C * D
    mean_crop_sample = np.mean(crop_sample, axis=(2, 3))
   
    return mean_crop_sample
    

def cropcenterScale_mean_4Sample(station_sample, cropScale, EC_center):
    # origin C * D * H * W → C * H * W
    # crop C * H * W →  C * Hs * Ws
    curr_loc_l = EC_center - ceilHalf(cropScale)
    curr_loc_r = EC_center + ceilHalf(cropScale)
    curr_loc_t = EC_center + ceilHalf(cropScale)
    curr_loc_b = EC_center - ceilHalf(cropScale)
    #
    crop_sample = station_sample[:,-1, curr_loc_l:(curr_loc_r+1), curr_loc_b:(curr_loc_t+1)]
    #
    mean_crop_sample = np.mean(crop_sample, axis=(1, 2))
    return mean_crop_sample


def cropECPrep_mean_4Sample(station_array, cropScale, EC_center):
    # H * W
    EC_origin_prep = station_array.values[-1,-1,:,:]
    # crop H * W →  Hs * Ws
    curr_loc_l = EC_center - ceilHalf(cropScale)
    curr_loc_r = EC_center + ceilHalf(cropScale)
    curr_loc_t = EC_center + ceilHalf(cropScale)
    curr_loc_b = EC_center - ceilHalf(cropScale)
    #
    crop_EC_origin_prep = EC_origin_prep[curr_loc_l:(curr_loc_r+1), curr_loc_b:(curr_loc_t+1)]
    #
    originPrepForecast = np.mean(crop_EC_origin_prep, axis=(0, 1)) * 1000
    return originPrepForecast

def floorHalf(x):
    floors = math.floor((x - 1) / 2)
    return floors


def cal_MAE(predicted, labels):
    mae = np.mean(np.abs(predicted - labels))
    return mae


def cal_TS_score(predicted, labels, threshold):
#     print ('predicted:',predicted)
#     print ('labels:',labels)
    labels = labels.squeeze()
#     print ('predicted:', predicted)
#     print ('label:', labels)
#     print ('threshold:', threshold)
#     print ('predicted_shape:', predicted.shape)
#     print ('label_shape:', labels.squeeze().shape)
    
    
    tp = np.sum((predicted >= threshold) * (labels >= threshold))
#     tn = np.sum((predicted < threshold) * (labels < threshold))
    fp = np.sum((predicted >= threshold) * (labels < threshold))
    fn = np.sum((predicted < threshold) * (labels >= threshold))
    #
    ts = round(tp / (tp + fp + fn), 5)
    return ts


def cal_PMAE(predicted, labels):
    labels = labels.squeeze()
    gaps = np.abs(predicted - labels)
    print ('predicted:', predicted)
#     print ('label:', labels)
#     print ('predicted_shape:', predicted.shape)
#     print ('label_shape:', labels.squeeze().shape)
    ##
    pCount = np.sum(labels > 1)
    pError = np.sum((labels > 1) * gaps)
    
    mpae = round(pError / pCount, 5)
    # luo
    # std
    return mpae


# def calStd4criteria1(con_cr_ele):
#     ele = []
#     for i,e in enumerate(con_cr_ele):
#         if e!=0:
#             ele.append(e)
#         else:
#             continue
#     ele = np.array(ele)
#     std_cri = np.std(ele)
    
    return std_cri 
    

'''
单卡方法
'''
def splitBatch(featuresArr, labelsVec, batchSize, seq, flag, epoch):
    bIdx = 0
    batch_data = []
    feature_size = featuresArr.shape[1]
#     ND * C
    if flag=='Train':
        # shuffle
        if epoch>0:
            # ND * C → N* DC 
            featuresArr_N = featuresArr.reshape(-1, seq*feature_size)
            # N* (DC+1)
            labelsVec_len = labelsVec.shape[0]
            total_data = np.concatenate((featuresArr_N, labelsVec.reshape(labelsVec_len, 1)),1)
            np.random.shuffle(total_data)
            featuresArr_update = total_data[:,:-1]
            featuresArr_update = featuresArr_update.reshape(-1, seq, feature_size).transpose((1,0,2))
            labels_update = total_data[:, -1]
        #
        if epoch==0:
           # D * N * C
            featuresArr_update = featuresArr.reshape(-1, seq, feature_size).transpose((1,0,2))
            labels_update = labelsVec
        #
        featuresArr_update = torch.from_numpy(featuresArr_update.astype(np.float32)).cuda()    
        labels_update = torch.from_numpy(labels_update.astype(np.float32)).cuda()
        train_iteration_num = math.floor(featuresArr_update.shape[1]/batchSize)
        # drop last
        for i in range (train_iteration_num):
            features_curr = featuresArr_update[:,bIdx:(bIdx+batchSize),:]
            labels_curr = labels_update[bIdx:(bIdx+batchSize)]
            batch_data.append((features_curr, labels_curr))
            bIdx = bIdx + batchSize
            # 注意越界
                 
    else:
        featuresArr_update = featuresArr.reshape(-1, seq, feature_size).transpose((1,0,2))
        featuresArr_update = torch.from_numpy(featuresArr_update.astype(np.float32)).cuda()
        #
        labels = torch.from_numpy(labelsVec.astype(np.float32)).cuda()
        test_iteration_num = math.floor(featuresArr_update.shape[1]/batchSize)
        # drop last
        for i in range (test_iteration_num):
            features_curr = featuresArr_update[:,bIdx:(bIdx+batchSize),:]
            labels_curr = labels[bIdx:(bIdx+batchSize)]
            batch_data.append((features_curr, labels_curr))
            bIdx = bIdx + batchSize
            
#         print ('features:', batch_data[0][0])
#         print ('data_length:', len(batch_data))

    return batch_data

def cal_CE(predict_probs, labels):
    # one-hot
    labels_oneHot = []
    n_class = predict_probs.shape[1]
    one_hot_trigger = np.eye(n_class, n_class)
    print ('preds_probs_shape:', predict_probs.shape)
    labels = labels.astype(np.int64)
    print ('labels:', labels)
    # labelling n*n_class
    for label in labels:
        if label>=n_class:
            label = n_class-1
        labels_oneHot.append(one_hot_trigger[label])    
#     labels_oneHot = [one_hot_trigger[label] for label in labels] 
    # muti-cross-entropy loss
    ce_samples = [-np.sum(labels_oneHot[idx]*predict_probs[idx]) for idx in range(len(predict_probs))]
    ce = round(np.sum(ce_samples)/predict_probs.shape[0],2)
    print ('cross-entropy:', ce)
    return ce


def cal_TS_score4Bayes(predicted, labels, bias_threshold):
    # 设定一个类bias和3个阈值，如果预测的类和标记类减去阈值在bias内，则tp+1
    # 如果预测类的bias>标记类的bias 空报，fn+1
    # 如果预测类的bias<标记类的bias 漏报，fp+1
    # 注意阈值的选择
    '''
    ???还是有问题，这个问题可以溯源到之前的回归问题如何输出一个需要的置信度的问题？？？
    值得深思！！！
    '''
    bias_pred_gt = abs(predicted-labels)
    
    tp = np.sum((bias_pred_gt == bias_threshold))
    fp = np.sum((bias_pred_gt > bias_threshold))
    fn = np.sum((bias_pred_gt < bias_threshold))
    #
    ts = round(tp / (tp + fp + fn), 5)
    return ts


def ceilHalf(x):
    ceils = math.ceil((x - 1) / 2)
    return ceils
      
def floorHalf(x):
    floors = math.floor((x - 1) / 2)
    return floors


##△ ycluo1
class ycluo1_dataLoader(data.Dataset):
    def __init__(self, cropDataFileNamesList, transform):
        self.cropDataFileNamesList = cropDataFileNamesList
        self.transformer = transform
        #
        self.loadCropData()
    
    def loadCropData(self):
        self.cropDataList = []
        for pickleFileName in tqdm.tqdm(self.cropDataFileNamesList, total=len(self.cropDataFileNamesList),
                                        desc="Load {} Data".format("luo1"), ncols=80, leave=False):
            cropDataDict = self.openPickleFile(pickleFileName)
            #
            for cropDataArray in cropDataDict.values():
                micapsValue = cropDataArray.attrs["micapValues"]

                if micapsValue < 0.1:
                    continue

                cropData = (cropDataArray.values, micapsValue)
                self.cropDataList.append(cropData)
                

    def openPickleFile(self, pickleFileName):
        if not os.path.exists(pickleFileName):
            raise NameError("No %s file exists." % (pickleFileName))

        with open(pickleFileName, 'rb') as f:
            oneCropData = pickle.load(f)

        return oneCropData

    def __len__(self):
        return len(self.cropDataList)

    def __getitem__(self, idx):
        oneData = self.cropDataList[idx]
        features = torch.from_numpy(oneData[0].astype(np.float32))
        micapsValue = np.array([oneData[1]], dtype=np.float32)

        if self.transformer:
            features = self.transformer(features)
            
        return features, micapsValue
    
    
#△ ycluo2  获取位置信息(ID站点需要对其)，对其站点然后再滑动截取窗口
class ycluo2_dataLoader(data.Dataset):
    def __init__(self, cropDataFileNamesList, transform, ts_len):
        self.cropDataFileNamesList = cropDataFileNamesList
        self.transformer = transform
        #
        self.LoadAlignStation4Data()
        self.loadCropData()
        
    ## 
    def LoadAlignStation4Data(self):
        fixstation_path = '/mnt/pami14/yqliu/dataset_meteo/stationItem.npy'
        latLonItem = np.load(fixstation_path)
        self.fix_station_indice = [tri[0] for tri in latLonItem]
        print ('fix_station_indice:', self.fix_station_indice)
    
    def loadCropData(self):
        #
        micapsValue_l = []
        cropData_l = []
        ts_count = 0
        cropData_ts_cand = []
        self.cropDataList = []
        
        for pickleFileName in tqdm.tqdm(self.cropDataFileNamesList, total=len(self.cropDataFileNamesList),
                                        desc="Load {} Data".format("luo2"), ncols=80, leave=False):
            cropDataDict = self.openPickleFile(pickleFileName)
            cropData_l = []

            #  根据station key → xarray 并且以3为TS length
            for station_id_key in self.fix_station_indice:
                if station_id_key not in np.array(cropDataDict.keys()):
                    continue
                #
                cropDataArray = cropDataDict[station_id_key]
                micapsValue = cropDataArray.attrs["micapValues"]
                #
                micapsValue_l.append(micapsValue)
                cropData = (cropDataArray.values, micapsValue, station_id_key)
                cropData_l.append(cropData)    
            ##
            print ('lenth_cropData_l:', len(cropData_l))
            cropData_ts_cand.append(cropData_l)
            ts_count = ts_count + 1
            
            if ts_count!=0 and ts_count%3==0:
                for crop_idx in range(len(cropData_l)):
                    
                    
                    
                
                    self.cropDataList
                 
                    
                if micapsValue < 0.1:
                    continue               
                
            
                
    def openPickleFile(self, pickleFileName):
        if not os.path.exists(pickleFileName):
            raise NameError("No %s file exists." % (pickleFileName))

        with open(pickleFileName, 'rb') as f:
            oneCropData = pickle.load(f)

        return oneCropData

    def __len__(self):
        return len(self.cropDataList)

    def __getitem__(self, idx):
        oneData = self.cropDataList[idx]
        features = torch.from_numpy(oneData[0].astype(np.float32))
        micapsValue = np.array([oneData[1]], dtype=np.float32)

        if self.transformer:
            features = self.transformer(features)
            
        return features, micapsValue
    
    




            
            
            
