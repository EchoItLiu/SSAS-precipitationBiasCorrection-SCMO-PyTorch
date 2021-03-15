import os
import datetime
import subprocess

# bzip2 -d FileName.bz2   解压bz2文件FileName,  并删除被解压的的.bz2原文件~~~

root_path = "/mnt/pami14/DATASET/METEOROLOGY/ECMWF2018"

# start_date = datetime.datetime.strptime("201801", "%Y%m%d")
# end_date = datetime.datetime.strptime("201811", "%Y%m%d")

# fileNameList = []
# month file names

# 2018-01 ** 需多处理一个文件，且多解压0104的月份
root_monthFile_list = os.listdir(root_path)[0:1]

# 2018-12做法 *** 处理方法同0104一样 
# root_monthFile_list = ['2018-12']

print (root_monthFile_list)


# 月份文件
for i, monthFileName in enumerate(root_monthFile_list):
    
    monthFileName = os.path.join(root_path, monthFileName)
    # all days
    day_allFile_list = os.listdir(monthFileName)
    
    # ** 处理0104.zip文件
    day_zipFile_list = list(filter(lambda s:isinstance(s,str) and len(s) > 4 ,day_allFile_list))
    
    # ** 处理0103文件夹
#     day_zipFile_list = ["0103"]
    
    print (day_zipFile_list)
    
    # 日文件
    for j, dayZipFile in enumerate(day_zipFile_list):
        
        dayZipPath = os.path.join(root_path, monthFileName, dayZipFile)
        dayFilePath = os.path.join(root_path, monthFileName, dayZipFile[:-4])
        
#         print (dayZipPath)
#         print (dayFilePath)


        # 解压缩 **0104.zip文件
        if not os.path.exists(dayFilePath): 
            
            pmz = subprocess.Popen(['sudo', 'unzip', '-d', monthFileName, dayZipPath], stdout = subprocess.PIPE)        
            out_dayFile = pmz.stdout.read().decode("utf-8").split()        
#             print (out_dayFile) 
            
            pmd = subprocess.Popen(['sudo', 'rm', '-rf', dayZipPath], stdout = subprocess.PIPE)        
            print ('{} is removed...'.format(dayZipFile))
            
        
         # ** 0103文件夹直接跳出循环
            
#         if dayZipFile=='0103':
#             continue
        
        
        # **  0104.zip文件  
        bz2_00_12_Files_list = os.listdir(dayFilePath)
        
        #  ** 0103文件夹 
#         bz2_00_12_Files_list = os.listdir(dayZipPath)        

                            
        # 截取需要的00和12起始的时间戳的
        for k, bz2TimestampFileName in enumerate(bz2_00_12_Files_list):
            
#             include_flag = False 
            
            # 去除掉最后两位为11的00时刻的
            if bz2TimestampFileName[-6:-4] == "11":
                continue
            
            
            day_unZipFile = dayZipFile[:-4] 
#             print ('test:', day_unZipFile)
            
            day2_upZipFile_datetime = datetime.datetime.strptime(day_unZipFile, "%m%d") + datetime.timedelta(hours = 12 * 2)
            day2_upZipFile = day2_upZipFile_datetime.strftime("%m%d")
                                    
            timeStamp1_con = bz2TimestampFileName.find(day_unZipFile + "00" + "00" + day_unZipFile + "00") == len(bz2TimestampFileName) - 21
            timeStamp2_con = bz2TimestampFileName.find(day_unZipFile + "00" + "00" + day_unZipFile + "03") == len(bz2TimestampFileName) - 21
            timeStamp3_con = bz2TimestampFileName.find(day_unZipFile + "00" + "00" + day_unZipFile + "06") == len(bz2TimestampFileName) - 21
            timeStamp4_con = bz2TimestampFileName.find(day_unZipFile + "00" + "00" + day_unZipFile + "09") == len(bz2TimestampFileName) - 21
            
###            
            timeStamp5_con = bz2TimestampFileName.find(day_unZipFile + "12" + "00" + day_unZipFile + "12") == len(bz2TimestampFileName) - 21
            timeStamp6_con = bz2TimestampFileName.find(day_unZipFile + "12" + "00" + day_unZipFile + "15") == len(bz2TimestampFileName) - 21
            timeStamp7_con = bz2TimestampFileName.find(day_unZipFile + "12" + "00" + day_unZipFile + "18") == len(bz2TimestampFileName) - 21
            timeStamp8_con = bz2TimestampFileName.find(day_unZipFile + "12" + "00" + day_unZipFile + "21") == len(bz2TimestampFileName) - 21            
            timeStamp9_con = bz2TimestampFileName.find(day_unZipFile + "12" + "00" + day2_upZipFile + "00") == len(bz2TimestampFileName) - 21   
            
            
            if (timeStamp1_con or timeStamp2_con or
                timeStamp3_con or timeStamp4_con or
                timeStamp5_con or timeStamp6_con or
                timeStamp7_con or timeStamp8_con or
                timeStamp9_con):
                    
                bz2Path = os.path.join(root_path, monthFileName, day_unZipFile, bz2TimestampFileName)
                #  
                pt = subprocess.Popen(['sudo', 'bunzip2', '-d', bz2Path], stdout = subprocess.PIPE)       
                #
                out_timestamp = pt.stdout.read().decode("utf-8").split()
                print (out_timestamp)
                                                       
                # rename
                EC2Path = os.path.join(root_path, monthFileName, day_unZipFile, bz2TimestampFileName[:-4])
                
                EC2Path_rename = os.path.join(root_path, monthFileName, day_unZipFile, bz2TimestampFileName[-24:-4])
                
                prename = subprocess.Popen(['sudo', 'mv', EC2Path, EC2Path_rename], stdout = subprocess.PIPE)
                                                                       
#                 include_flag = True 
                
#             include_l.append(include_flag)
            
        bz2_00_12_Files_list_update = os.listdir(dayFilePath)
                                                                                                    
        bz2_removing = list(filter(lambda s:isinstance(s,str) and len(s) == 55, bz2_00_12_Files_list_update))
        
        print (bz2_removing)
        
        
        for removing in bz2_removing: 
            
            ba2_removeing_path = os.path.join(root_path, monthFileName, day_unZipFile, removing)
            incdel_f = subprocess.Popen(['sudo', 'rm', '-rf', ba2_removeing_path], stdout = subprocess.PIPE)
            print ('{} is removed...'.format(removing))
                    
                
                    
                                                                                           