import os
import sys
import tqdm
import math
import pygrib
import pickle
# import argparse
import datetime
import time
import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate
from dateutil.relativedelta import relativedelta
import math


normalFactors = ["Relative humidity",
              "Vertical velocity",
              "Temperature",
              "V component of wind",
              "U component of wind",
              "Potential vorticity",
              "Divergence",
              "Geopotential Height"
              ]


singleLevelFactors = ["Total column water vapour",
                   "Total column water",
                   "Total cloud cover"]

# initialTimeLevelFactors =  ["Convective available potential energy",
#                    "Total column water",
#                    "Total cloud cover"]


#  ***  Convective precipitation ***
precipitationFactors = ["Large-scale precipitation",
                     "Convective precipitation",
                     "Total precipitation"]

# Level
# levelsList = [500, 700, 850, 925]
levelsList = [500, 700, 850]

# Eastern Region
east_locBound = (20, 40, 110, 124)

# Value Range from Eastern Region (Enlarge 2°)
# from top to bottom, from left to right
sw_Bound_lat = np.arange(east_locBound[0] - 2,east_locBound[1] + 2 + 0.125, 0.125)[::-1]
sw_Bound_lon = np.arange(east_locBound[2]- 2,east_locBound[3] + 2 + 0.125, 0.125)

# interpolate range (refined and coarse)
refined_lat = np.arange(0, east_locBound[1] - east_locBound[0] + 4 + 0.125, 0.125)
refined_lon = np.arange(0, east_locBound[3] - east_locBound[2] + 4 + 0.125, 0.125)

coarse_lat = np.arange(0, east_locBound[1] - east_locBound[0] + 4 + 0.25, 0.25)
coarse_lon = np.arange(0, east_locBound[3] - east_locBound[2] + 4 + 0.25, 0.25)

# setting
EC_path ="/mnt/pami14/DATASET/METEOROLOGY/ECforecast/GRIB"

GTmicaps_path = "/mnt/pami14/DATASET/METEOROLOGY/micaps_16_18"

TS_path = "/mnt/local2/yqliu/ec_ts_modals"

utility_path = "/mnt/pami14/yqliu/EC-BiasCorrecting/SHO/util_dataset"

errors_path = "/mnt/pami14/yqliu/biasCorrection_II/errors_text"

# 太长，数据过大，LSTM也记不住,
ts_length = 6

slidingBar_stride = 1

scope_slidingWindows =  0.125 * 1
# h × w = 29 * 29
scope_stationWindows = 28

# argument for discarding non-rain
discard_rainless_ratio = 0.8
estimate_average_rainless_num = 600

# sampling frequency for nonrain 0 | 0<p<0.1
# 这两个变量用来严格控制0样本和0<p<0.1的样本的数量
sample_freq0_4rain = 0.16
sample_freq0_01_4rain = 0.8


class TS_ec(object):
    def __init__(self, ecTuple):   
        self.ecFilePath = ecTuple
        
    def ts_ec_selection(self):  
        nc2, nc3_1, nc4_2, pc1, pc2, pc3, pc4 = self.ec_Select()
        nc_2_ec_features = self.ec_Data(nc2, 0, pc2, True)
        nc3_1_ec_features = self.ec_Data(nc3_1, pc1, pc3, False)
        nc4_2_ec_features = self.ec_Data(nc4_2, pc2, pc4, False)   
        return nc_2_ec_features, nc3_1_ec_features, nc4_2_ec_features
            
                    
    def ec_Select(self):
        ## open 5 EC files
        # zero(0/12) 
        init_ec_mes = pygrib.open(self.ecFilePath[0])
        # 3 / 15
        ec_1_mes = pygrib.open(self.ecFilePath[1]) 
        # 6 / 18
        ec_2_mes = pygrib.open(self.ecFilePath[2]) 
        # 9 / 21       
        ec_3_mes = pygrib.open(self.ecFilePath[3])
        # 12 / 00
        ec_4_mes = pygrib.open(self.ecFilePath[4])
        
            
        
        ## select 5 EC Classes according to keywords of factors     
        # normal Factors + singleLevelFactors
        
        print ('........ec class reading start........')
        before_time_1 = time.time()
        nf_class_init = [init_ec_mes.select(name = keyFactor,level = keyLevel)[0] for keyFactor in normalFactors for keyLevel in levelsList] + [init_ec_mes.select(name = keyFactor)[0] for keyFactor in singleLevelFactors] 
        print ('-------over-one-------')
        
        nf_class_1 = [ec_1_mes.select(name = keyFactor,level = keyLevel)[0] for keyFactor in normalFactors for keyLevel in levelsList] + [ec_1_mes.select(name = keyFactor)[0] for keyFactor in singleLevelFactors]
#         print ('-------over-two-------')
        
        nf_class_2 = [ec_2_mes.select(name = keyFactor,level = keyLevel)[0] for keyFactor in normalFactors for keyLevel in levelsList] +  [ec_2_mes.select(name = keyFactor)[0] for keyFactor in singleLevelFactors]
        
#         print ('-------over-three-------')

        nf_class_3 = [ec_3_mes.select(name = keyFactor,level = keyLevel)[0] for keyFactor in normalFactors for keyLevel in levelsList] + [ec_3_mes.select(name = keyFactor)[0] for keyFactor in singleLevelFactors]
        
#         print ('-------over-four-------')       
        nf_class_4 = [ec_4_mes.select(name = keyFactor,level = keyLevel)[0] for keyFactor in normalFactors for keyLevel in levelsList] + [ec_4_mes.select(name = keyFactor)[0] for keyFactor in singleLevelFactors]
        after_time_1 = time.time()
#         print ('-------over-five-------')
          ## Class Cost Time : 190s  
#         print ('ec_class_cost_time:', after_time_1 - before_time_1)
        
        
        # append normal Factors between 6h interval
        nf_class_2 = nf_class_init  + nf_class_2
        nf_class_3_1 = nf_class_3 + nf_class_1 
        nf_class_4_2 = nf_class_4  + nf_class_2
        
        # assign normal factors between all combined factors 
        min_channel = min(len(nf_class_2),len(nf_class_3_1),len(nf_class_4_2))
        nf_class_2 = nf_class_2[:min_channel]
        nf_class_3_1 = nf_class_3_1[:min_channel]
        nf_class_4_2 = nf_class_4_2[:min_channel]
        
        print ('min_channel:', min_channel)
        
        # precipitation
        prep_class_1 = [ec_1_mes.select(name = keyFactor)[0] for keyFactor in precipitationFactors]
        prep_class_2 = [ec_2_mes.select(name = keyFactor)[0] for keyFactor in precipitationFactors]
        prep_class_3 = [ec_3_mes.select(name = keyFactor)[0] for keyFactor in precipitationFactors]
        prep_class_4 = [ec_4_mes.select(name = keyFactor)[0] for keyFactor in precipitationFactors]
        
        
        ## Normal Factors data according to Eastern Region
        return nf_class_2, nf_class_3_1, nf_class_4_2, prep_class_1, prep_class_2, prep_class_3, prep_class_4 
        
    
    def ec_Data(self, nc_interval, before_pc, after_pc, flag_0):
        ec_features  = []
        before_time_2= time.time()
        
        print ('........ec data reading start........')
        for nc_grib in nc_interval:
            grib_data = nc_grib.data(lat1 = east_locBound[0] - 2, lat2 = east_locBound[1] + 2, lon1 = east_locBound[2] - 2, lon2 = east_locBound[3] + 2)[0]
          
            if grib_data.shape[0]< len(refined_lat):
                
                grib_data = interpolate.interp2d(coarse_lon, coarse_lat, grib_data, kind='linear')( refined_lon, refined_lat) 
        
            ec_features.append(grib_data)
        
        if flag_0: 
            before_prep_date = before_pc
        
        else:               
            before_prep_date = [bf_pc.data(lat1 = east_locBound[0] - 2, lat2 = east_locBound[1] + 2, lon1 = east_locBound[2] - 2, lon2 = east_locBound[3] + 2)[0] * 1000 for bf_pc in before_pc]
                   
        after_prep_date = [af_pc.data(lat1 = east_locBound[0] - 2, lat2 = east_locBound[1] + 2, lon1 = east_locBound[2] - 2, lon2 = east_locBound[3] + 2)[0] * 1000 for af_pc in after_pc]
        
        
        if flag_0:         
            interval_prep_date = after_prep_date
            
        else:
            interval_prep_date = [inv for inv in (map(lambda x: x[1] - x[0], zip(before_prep_date, after_prep_date)))]
            
        ec_features += interval_prep_date
               
        # list → numpy N(30 * 2 * 3 = 180) * C * H * W
        ec_features = np.stack(ec_features)
        after_time_2= time.time()
        ## Data Average Cost Time : 16s         
        print ('ec_data_cost_time:', after_time_2 - before_time_2)
        return ec_features
    
def PreECF(preTimeStamp):
    # init
    ecPath = EC_path
    start_time = preTimeStamp
    ec_features_l = []
    ec_features_path_l = []
    test_shape = []
    
    print ('---')

    if isinstance(preTimeStamp, str):
        # per one month
        start_time = datetime.datetime.strptime(start_time, "%Y%m%d")
        end_time = start_time + relativedelta(months = 1)        
        ecFiles_list = [start_time + datetime.timedelta(hours=12 * x) for x in range(0, 2 * (end_time - start_time).days)]
#         ecFiles_list = [start_time + datetime.timedelta(hours=12 * x) for x in range(0,8)]
        
           
    for idx, ecFile in tqdm.tqdm(
            enumerate(ecFiles_list), total=len(ecFiles_list),
            desc='Load ECPath', ncols=80,
            leave = True):
        yearStr = ecFile.strftime("%Y")
        mdhStr = ecFile.strftime("%m%d%H")
        ec_dir_name = os.path.join(ecPath, yearStr, mdhStr)
        
        if not os.path.exists(ec_dir_name):
            continue
        
        # 
        if yearStr == "2017":
            
            if mdhStr== "051200" or mdhStr == "051312" or mdhStr== "083012" or mdhStr=="100412":
                print ("C1D..files had been broken!")
                continue
            if  mdhStr=="071412" or mdhStr=="100300" or mdhStr=="102400":
                print ("These C1D files are not exist!")
                continue
                
                          
        """5 EC sliceFile per ecfile 0/3/6/9/12 || 12/15/18/21/0"""
        init_Date = ecFile.strftime("%m%d%H")   
        
        ec_1_Date = (ecFile + datetime.timedelta(hours = 3)).strftime("%m%d%H")
        ec_2_Date = (ecFile + datetime.timedelta(hours = 6)).strftime("%m%d%H")
        ec_3_Date = (ecFile + datetime.timedelta(hours = 9)).strftime("%m%d%H")
        ec_4_Date = (ecFile + datetime.timedelta(hours = 12)).strftime("%m%d%H")
       
        ## concat to EC sliceFile Name
        # #  zero
        init_ec_path = os.path.join(ec_dir_name, "C1D" + mdhStr + "00" + init_Date + "001")
        # 3 / 15
        ec_1_path = os.path.join(ec_dir_name, "C1D" + mdhStr + "00" + ec_1_Date + "001")
        # 6 / 18
        ec_2_path = os.path.join(ec_dir_name, "C1D" + mdhStr + "00" + ec_2_Date + "001")
        # 9 / 21
        ec_3_path = os.path.join(ec_dir_name, "C1D" + mdhStr + "00" + ec_3_Date + "001")
        # 12 / 00
        ec_4_path = os.path.join(ec_dir_name, "C1D" + mdhStr + "00" + ec_4_Date + "001")
        
        tsEC = TS_ec((init_ec_path, ec_1_path, ec_2_path, ec_3_path, ec_4_path))
        
        # ecf(2, 3_1, 4_2) ↔ ecpath(2, 3_1, 4_2)
        nc2EcFeatures, nc31EcFeatures, nc42EcFeatures = tsEC.ts_ec_selection()
                        
        ec_features_l.append(nc2EcFeatures)
        ec_features_l.append(nc31EcFeatures)
        ec_features_l.append(nc42EcFeatures)
        
                            
        ec_features_path_l.append(ec_2_path)
        ec_features_path_l.append(ec_3_path)
        ec_features_path_l.append(ec_4_path)
                        
    # list → numpy [N*C*H*W]
    ec_features = np.stack(ec_features_l)
      
    # list → numpy [N]
    ec_features_path = np.array(ec_features_path_l)
    return ec_features, ec_features_path 
        

def fixStation():
    micapsPath = GTmicaps_path
    ## select fixed 3 micaps file
    ## fixed_station_number: 1188  17080608.000  17070814.000  16081020.000"""
    candi_mi_path1 = os.path.join(micapsPath, "2017", "08", "surface", "r6-p", "17080608.000")
    candi_mi_path2 = os.path.join(micapsPath, "2017", "07", "surface", "r6-p", "17070814.000")
    candi_mi_path3 = os.path.join(micapsPath, "2016", "08", "surface", "r6-p", "16081020.000")

    candi_paths_l = [candi_mi_path1, candi_mi_path2, candi_mi_path3]
    # init 
    index_l = []
    lats_d = {}
    lons_d = {}

    for cPath in candi_paths_l:
        if not os.path.exists(cPath):
            print("candidate micaps not exists!")

        with open(cPath, encoding="GBK") as f:  
            cPath_data = f.readlines()
            key_data = cPath_data[14:]

        for oneLine in key_data:
            oneLabel = oneLine.split()
            index, lon, lat, _, _ = map(float, oneLabel) 

            # select strictly
            if (lat < east_locBound[0]) or (lat > east_locBound[1]):
                continue
            elif (lon < east_locBound[2]) or (lon > east_locBound[3]):
                continue
            else:
                index_l.append(index)                       
                lats_d[index] = lat        
                lons_d[index] = lon

    # remove redundancy and fixed indice      
    index_fixed = np.unique(np.array(index_l))     
    # get lat/lon pairs [(index, lat, lon),...]
    lats_lons_item = [(fIdx, lats_d[fIdx], lons_d[fIdx]) for fIdx in index_fixed]
    return lats_lons_item

def convertPress_nor_flag(st_press):
    if st_press!=9999:
        st_press = float(st_press/10)
        if abs(round(st_press,2)+1000-1000)<abs(round(st_press,2)+900-1000):
            origin_st_press = round(st_press+1000,2)
        else:
            origin_st_press = round(st_press+900,2)

        convert_st_press = round(origin_st_press - (1000+900)/2,2)
     
    # maintain 9999
    else:
        convert_st_press = st_press
    return convert_st_press
    
           
class pre_ts_ecData(object):
    def __init__(self, ecFeatures, ecFeaturePaths, latLonEntries, monthTimeStamp):
        self.savePath = TS_path
        self.micapsPath = GTmicaps_path
        self.errorPath = errors_path
        self.latlons_item = latLonEntries
        self.ec_Features = ecFeatures
        self.ec_FeaturePaths = ecFeaturePaths
        self.monthTimeStamp = monthTimeStamp 
        self.scopeSW = scope_slidingWindows
        self.scopeGW = scope_stationWindows
        self.tsLen = ts_length
        self.slidB = slidingBar_stride
        self.ec_features_ts_l = []
        self.ec_features_path_ts = []
        self.sliceWindows4LatLon()
        self.slidingTSec()
        self.gainGT4ecPath()
  
    def sliceWindows4LatLon(self):
        ec_features_stations_l = []
        update_lats_lons_item = []
        update_indice = []
        
        # station windows range → h × w
        self.gw_range = math.floor(self.scopeGW / 2)
        
        for k in tqdm.tqdm(
                range(len(self.latlons_item)), total=len(self.latlons_item),
                desc='Load StationWindows', ncols=80,
                leave = True):
            flag = False
            for i, lat in enumerate(sw_Bound_lat):
                if flag == True:
                    break
                for j, lon in enumerate(sw_Bound_lon):
                    # sliding window scope
                    
                    lat_range = (lat - self.scopeSW, lat)
                    lon_range = (lon, lon + self.scopeSW)  
                    
                    # search and matching
                    if self.latlons_item[k][1]<=lat_range[1] and self.latlons_item[k][1]>=lat_range[0] \
                    and self.latlons_item[k][2] <= lon_range[1] and self.latlons_item[k][2]>= lon_range[0]:
                    # slice station windows - from bottom to top ,and from left to right 
                    # windows shape - [self.scopeGW * self.scopeGW] N * C * H *W
                        ec_features_stations_l.append(self.ec_Features[:,:,(i - self.gw_range):(i + self.gw_range + 1), (j - self.gw_range):(j + self.gw_range + 1)])
                    # save existing station info.
                        update_indice.append(self.latlons_item[k][0])
                        
                        flag = True
                        break
                        
                                  
        # list → numpy n * N * C * h * w (float32) 
        self.ec_features_stations = np.stack(ec_features_stations_l).astype(np.float32)        
        print ('EC_FEATURES_STATIONS:', self.ec_features_stations.shape)
        
        # n 
        self.update1_indice  = update_indice
        
        if len(self.update1_indice)< 1188:
            errorStr = 'Update Stations is Inconsistent!\n'
            with open(os.path.join(self.errorPath, "error_log.txt"), "a+", encoding='utf-8') as f:
                f.write(errorStr)    

                
        ### saving Xarray to one file
        # output: ec_features_s --  n * N * C * h * w
        # update_indice  n
        # ecFeaturePaths  N
        # C random for aligning
        # h random for aligning
        # w random for aligning                   
              
    def slidingTSec(self):

        b_idx = 0
        e_idx = self.tsLen - 1
        self.ec_features_ts = []
        self.ec_features_path_ts = []
        l_count = 0
        # N * n * C * h * w
        ec_features_s = np.transpose(self.ec_features_stations,(1,0,2,3,4))
        
        while (e_idx) <= (len(ec_features_s)-1):
            if l_count == 0 and e_idx > (len(ec_features_s)-1):
                print ("ec length is smaller than time-series sequence. please reset argument!")
                break
               
            self.ec_features_ts.append(ec_features_s[b_idx:(b_idx + self.tsLen)])
            # [[..], [..], [..]]
            self.ec_features_path_ts.append(self.ec_FeaturePaths[b_idx:(b_idx + self.tsLen)])

            b_idx = b_idx + self.slidB
            e_idx = e_idx + self.slidB
            l_count = l_count + 1
            
        # L * D * n * C * h * w  /  L 
        # one sample: C * D * h * w FEATURES - 1 GT
        # one sample: C * D * h * w FEATURES - (D-τ) GT must unify stationID
        print ('EC_FEATURES_ALL_SEQUENCE:', len(self.ec_features_ts))
        
        
    def getECTsDate(self, file_Names):
        ECTsDate = [dataPath.split("/")[-1] for dataPath in file_Names]
        return ECTsDate
        
        
       
    def gainGT4ecPath(self):        
        count_total_samples = 0
        count_rain_samples = 0
        
        # month path
        current_pkl_list = os.listdir(self.savePath)
        
        # file series
        if current_pkl_list==[]:
            seriesNo = 0

        if current_pkl_list!=[]:
            seriesNo = 0

            for PrepklName in current_pkl_list:
                PrepklTime = datetime.datetime.strptime(PrepklName[-6:], "%Y%m")
                PostpklTime = datetime.datetime.strptime(self.monthTimeStamp[:-2], "%Y%m")

                if (PostpklTime - PrepklTime)>=datetime.timedelta(days=31) or \
                  (PostpklTime - PrepklTime)>=datetime.timedelta(days=30):
                    seriesNo = seriesNo + 1 
                                       
        month_path = "{}-modals-seq".format(seriesNo) + self.monthTimeStamp[:-2]
                        
        count_total_samples = 0
        count_rain_samples = 0
        count_rainless_samples = 0
        total_rainSamples0 = 0
        total_rainSamples0_01 = 0
       
        for lth, fileNames in tqdm.tqdm(
                enumerate(self.ec_features_path_ts), total=len(self.ec_features_path_ts),
                desc='Load One Data', ncols=80,
                leave = True):

            ## init
            # D * n * C * h * w →  n * C * D * h * w 
            ec_features_ts_lth = self.ec_features_ts[lth].transpose(1,0,2,3,4).transpose(0,2,1,3,4)
            ec_ts_last = fileNames[-1].split("/")[-1]
            
            ec_ts_dates = self.getECTsDate(fileNames)
#             print ('ec_ts_last_path:', ec_ts_last)
#             print ('ec_ts_last_path_type:', type(ec_ts_last))
            
            xar_dic = {}
            xar_dic_update = {}
            update_ec_features_ts_l = []
            modals_filenames = []
            mi_filenames = []
            indice_all = []
            # count for ts last rain value
            count_0rain_samples = 0
            count_0_01_rain_samples = 0
                                 
            # calcuate common stations between all sequential diamond 3
            for fileName in fileNames:
                # init zero
                modal_dic = {}
                mi_dic = {}
                indice = []
                
                fileNameSplit = fileName.split("/")
                fileNameStr = fileNameSplit[-1]
                oldMdhStr = fileNameStr[-9:-3]
                oldYearStr = fileNameSplit[-3]

                labelDateStr = oldYearStr + oldMdhStr
                # convert BJ time
                labelDate = datetime.datetime.strptime(labelDateStr, "%Y%m%d%H") + datetime.timedelta(hours=8)
                yearStr = labelDate.strftime("%Y")
                monthStr = labelDate.strftime("%m")

                modalsDirPath = os.path.join(self.micapsPath, yearStr, monthStr, "surface", "plot")
                labelDirName = os.path.join(self.micapsPath, yearStr, monthStr, "surface", "r6-p")

                labelFileName = datetime.datetime.strftime(labelDate, "%Y%m%d%H")[2:] + ".000"

                modalsFileFullName = os.path.join(modalsDirPath, labelFileName)
                labelFileFullName = os.path.join(labelDirName, labelFileName)


                # extract diamond 1
                if not os.path.exists(modalsFileFullName):
                    print("diamond 1 File is not exist!")
                    continue 
                    
                with open(modalsFileFullName, encoding="GBK") as f:
                    
                    cPath_data = f.readlines()        
                    key_data = cPath_data[2:]
                 
                for i, oneLine in enumerate(key_data):
                    # pre+post才是一行完整数据
                    if (i+1)%2==0:
                        post_label = oneLine.split()                    
                        oneLabel = pre_label + post_label 
                        
                        index = int(oneLabel[0])
                        lon = float(oneLabel[1])
                        lat = float(oneLabel[2])                            
                        wind_direct = float(oneLabel[6])
                        wind_velocity = float(oneLabel[7])
                        # conver and nor
                        pressure = float(convertPress_nor_flag(float(oneLabel[8])))
                        # humidness
                        dew_point = float(oneLabel[16])
                        temperature = float(oneLabel[19])
                        
                        modal_dic[index] = (lon, lat, temperature, pressure, wind_direct, wind_velocity, dew_point)    
                    pre_label = oneLine.split()
                    
                                   
                # 按时间序列从左向右依次append多模态数据字典
                ## [{t-2},{t-1},{t},...]
                modals_filenames.append(modal_dic)
                                    
                # extract diamond 3
                if not os.path.exists(labelFileFullName):
                    print("diamond 3 File is not exist!")
                    continue
                    
                with open(labelFileFullName, encoding="GBK") as f:
                    mData = f.readlines()
                    micaps_data = mData[14:]

                for oneLine in micaps_data:
                    oneLabel = oneLine.split()
                    index, lon, lat, _, value = map(float, oneLabel)
                    mi_dic[index] = (value,lat,lon)
                    indice.append(index)
                    
                # 按时间序列从左向右依次append降水值
                ## [{t-2},{t-1},{t},...]      
                mi_filenames.append(mi_dic)
                ## 每个序列元素的站点
                indice_all.append(indice)
                
            ## 序列站点对齐
            index_v1 = indice_all[0]
            index_v2 = indice_all[1]
            index_v3 = indice_all[2]
            index_v4 = indice_all[3]
            index_v5 = indice_all[4]
            index_v6 = indice_all[5]
            
            # drop last
            if len(indice_all)==6:
                index_v12 = [v11 for v11 in index_v1 for v22 in index_v2 if v11==v22]
                index_v123 = [v33 for v12 in index_v12 for v33 in index_v3 if v12==v33]
                index_v1234 = [v44 for v123 in index_v123 for v44 in index_v4 if v123==v44]
                index_v12345 = [v55 for v1234 in index_v1234 for v55 in index_v5 if v1234==v55]
                indice_inter = [v66 for v12345 in index_v12345 for v66 in index_v6 if v12345==v66]
                
                indice_TPDWP = [v_prep for v_prep in self.update1_indice for v_inter in indice_inter if v_prep==v_inter]
                print ('TPDWP_STATIONS_NUMS:', len(indice_TPDWP))
                
            # last ts    
            else:
                continue
                
                
                     
            # t-τ → t sequence
            # 根据站点来截取每个多模态样本数据，并append所有时间序列的多模态labels
            for v, up_index in enumerate(self.update1_indice): 
                
                # alignment: no EC (ignore GT) or no micapsGT(ignore EC) 
                if up_index in indice_inter:
                    # count 
                    if mi_filenames[-1][up_index][0]==0:                       
                        count_0rain_samples = count_0rain_samples + 1
                    if mi_filenames[-1][up_index][0]>0 and mi_filenames[-1][up_index][0]<0.1:
                        count_0_01_rain_samples = count_0_01_rain_samples + 1                                      
                    if mi_filenames[-1][up_index][0]>=0.1:  
                        count_rain_samples = count_rain_samples  + 1
                        
                    if mi_filenames[-1][up_index][0]<0.1:
                        count_rainless_samples = count_rainless_samples + 1
                    
                    count_total_samples = count_total_samples + 1                        
                                       
                    # length: D
                    seq_tempers = []
                    seq_press = []
                    seq_wind = []
                    seq_dew = []
                    
                    seq_gt_prep = []
                    seq_lon = []
                    seq_lat = []
                    
                    # modals
                    for modals_dic in modals_filenames:
                        # 
                        _, _, temperature, pressure, wind_direct, wind_velocity, dew_point = modals_dic[up_index]
                        seq_tempers.append(temperature)
                        seq_press.append(pressure)
                        seq_wind.append((wind_direct, wind_velocity))
                        seq_dew.append(dew_point)
                        
                    
                    for mi_dic in mi_filenames:
                        # 
                        gt_prep_value, lat_up, lon_up = mi_dic[up_index]
                        seq_gt_prep.append(gt_prep_value)
                        seq_lon.append(lon_up)
                        seq_lat.append(lat_up)
                                                           
                    # slice lat/lon range
                    lat_upper_bound = lat_up + self.gw_range * 0.125
                    lat_lower_bound = lat_up - self.gw_range * 0.125
                    lon_left_bound = lon_up - self.gw_range * 0.125
                    lon_right_bound = lon_up + self.gw_range * 0.125 
                                          
                    # from top to bottom , from left to right
                    lat_range = list(np.arange(lat_lower_bound, lat_upper_bound + 0.125, 0.125)[:29])
                    lat_range = np.array([round(i,2) for i in lat_range]).astype(np.float32)
                    lon_range = list(np.arange(lon_left_bound, lon_right_bound + 0.125, 0.125)[:29])
                    lon_range = np.array([round(i,2) for i in lon_range]).astype(np.float32)
                    
                    channel_num_l = np.arange(0, ec_features_ts_lth[v].shape[0], 1)
                    dynamic_num_l = np.arange(0, ec_features_ts_lth[v].shape[1], 1)

                    # create xarray -  C * D * h * w (float 32)
#                     print ('EC_FEATURE:', ec_features_ts_lth[v].shape)
                    
                    xar = xr.DataArray(ec_features_ts_lth[v].astype(np.float32), coords=[channel_num_l, dynamic_num_l, lat_range, lon_range], dims=['key_channel','seq_length','lat_range','lon_range'])
                    
                    # precipitation GT in sequence
                    xar.attrs['seq_prep_gt'] =  seq_gt_prep
                    xar.attrs['seq_tem_gt'] =  seq_tempers
                    xar.attrs['seq_press_gt'] = seq_press
                    xar.attrs['seq_wind_gt'] =  seq_wind
                    xar.attrs['seq_dew_gt'] =  seq_dew 
                    xar.attrs['ec_fileName'] = ec_ts_dates
                    # 
                    xar_dic[up_index] = xar
            
            # 保存序列文件名(时间戳)
            ## resampling for non-rain
            #根据采样率统计当前序列文件下的非降水值0和0<p<0.1的数量
            print ('sample_freq0_4rain:', sample_freq0_4rain)
            print ('sample_freq0_01_4rain:', sample_freq0_01_4rain)
            count_rainSamples0 = int(count_0rain_samples * sample_freq0_4rain)
            count_rainSamples0_01 = int(count_0_01_rain_samples * sample_freq0_01_4rain)
            # init current count
            count_0 = 0
            count_0_01 = 0
            
            for station_id, station_xarray in xar_dic.items():
                if station_xarray.attrs["seq_prep_gt"][-1]==0 and count_rainSamples0>=count_0:
                    xar_dic_update[station_id] = station_xarray
                    count_0 = count_0 + 1
                    
                if station_xarray.attrs["seq_prep_gt"][-1]>0 and station_xarray.attrs["seq_prep_gt"][-1]<0.1 and count_rainSamples0_01>=count_0_01:
                    xar_dic_update[station_id] = station_xarray
                    count_0_01 = count_0_01 + 1
                    
                if  station_xarray.attrs["seq_prep_gt"][-1]>=0.1:
                    xar_dic_update[station_id] = station_xarray
                    
           # test
            total_rainSamples0 = total_rainSamples0 + count_rainSamples0 
            total_rainSamples0_01 = total_rainSamples0_01 + count_rainSamples0_01   
            
        
            print ('count_rainSamples0:', count_rainSamples0)
            print ('count_rainSamples0_01:', count_rainSamples0_01)
            print ('count_0:', count_0)
            print ('count_0_01:', count_0_01)
                                                                                            
            
            ## save according to time level            
            # (n - τ) * C * D * h * w saving xarray, τ is uncertainty
            
            # file path
            '''
            pklFilePath = os.path.join(self.savePath, month_path)
            if not os.path.exists(pklFilePath):
                os.mkdir(pklFilePath)
                
            pkl_name = os.path.join(pklFilePath, fileNames[-1].split("/")[-3] + fileNames[-1].split("/")[-1][-9:-3]+"modals"+".pkl")
            
            print ('PKL_NAME:', pkl_name)

            with open(pkl_name,"wb") as f:
                pickle.dump(xar_dic_update, f, protocol = pickle.HIGHEST_PROTOCOL)
            '''
        
        print ('--'*20)
            
        print ('total_rainSamples0:', total_rainSamples0)
        print ('total_rainSamples0_01:', total_rainSamples0_01)            
        print ('TOTALsamples:', count_total_samples)
        print ('RAINsamples:', count_rain_samples)
        print ('RAINLESSsamples:', count_rainless_samples)                
                                                    

if __name__== "__main__":
    
#     timestamp_items = ['20160801']
    ec_features, ec_features_path = PreECF(sys.argv[1])
#     ec_features, ec_features_path = PreECF(timestamp_items[0])
    ## generate fixed station list
    fixstation_path = os.path.join(utility_path, "stationItem.npy")
    
    if not os.path.exists(fixstation_path): 
        # type: list
        latLonItem = fixStation()
        np.save(fixstation_path, latLonItem)
    else:
        # type: numpy ndarray
        latLonItem = np.load(fixstation_path)
    #
    pre_ts_ecData(ec_features, ec_features_path, latLonItem, sys.argv[1])
#     pre_ts_ecData(ec_features, ec_features_path, latLonItem, timestamp_items[0])
    afterOpentime = time.time()
    


