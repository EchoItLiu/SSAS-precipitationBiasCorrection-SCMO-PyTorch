import torch
import numpy as np
import yaml
import sys
sys.path.append("/home/yqliu/EC-BiasCorrecting")
sys.path.append("/home/yqliu/EC-BiasCorrecting/pre_training")
import os
import os.path as osp
from torch.utils import data
import torchvision
from datasets.SASDataset import modalGribDataset, modalOrdinalDataset
from utils.util_preTraining import updateMeBatch4Label, BP_ME
from utils.util_main import cropOneScale4EC, get_file_name, ceilHalf, floorHalf
import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch import nn
import copy
import argparse
from torchvision import transforms

'''
python pre_training/MSM_pretraining/P_MSM_Pre.py -c ./config/SHO.yaml -r 0.3 -d 5 6
'''
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', help='config file', type=str, required=True)
parser.add_argument('--dropRate', '-r', help='drop some samples randomly', type=float, required=True)
parser.add_argument('--device_ids', '-d', nargs='*', help='devices id to be used', type=int)
parser.add_argument('--learning_rate', help='devices id to be used', type=float, default=1e-4)
parser.add_argument("--num_workers", help='num_workers', default=16, type=int)
parser.add_argument('--in_channel', help='meteorological factor nums', type=int, default=57)
parser.add_argument("--basePath", default = "/mnt/pami14/yqliu/EC-BiasCorrecting")
parser.add_argument("--preEpoch", help='pretraining epoch', type=int, default=50)


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## params参数导出
with open(args.config) as f:
    config = yaml.load(f)
args.dataset_name = get_file_name(args.config)
## dic
params_SAS = config['SAS']
#
dataset_path = osp.join(args.basePath, args.dataset_name, params_SAS['dataset_type'])
SMS_save_path = osp.join(args.basePath, params_SAS['preTrain_MSM'])
#
prepRange = np.arange(params_SAS['set_prep_space'], params_SAS['set_max_prepValue'], params_SAS['set_prep_space'])
nClass = prepRange.shape[0]

## 选择MSM
def selectMSM(modelType):
    if modelType == 1:
        from pre_training.MSM import DeformConvNetI
        model = DeformConvNetI
    elif modelType == 2:
        from pre_training.MSM import DeformConvNetII
        model = DeformConvNetII
    elif modelType == 3:
        from pre_training.MSM import DeformConvNetIII
        model = DeformConvNetIII
    else:
        raise NotImplementedError       
    return model

def initMSM(element_flag):
    if element_flag=='tem':
        eModel = selectMSM(1)
        init = eModel(args.in_channel)
    if element_flag=='press':
        eModel = selectMSM(1)
        init = eModel(args.in_channel)
    if element_flag=='wind':
        eModel = selectMSM(2)
        init = eModel(args.in_channel)
    if element_flag=='dew':
        eModel = selectMSM(1)
        init = eModel(args.in_channel)        
    if element_flag=='rain':
        eModel = selectMSM(1)
        init = eModel(args.in_channel)        
    return init
        
## 5个预训练模型需要的子模块
press_MSM = initMSM('press')
# nnParallel
torch.cuda.set_device(args.device_ids[0])
if len(args.device_ids) > 1:
    press_MSM = nn.DataParallel(press_MSM, device_ids=args.device_ids).cuda() 
else:
    press_MSM = press_MSM.cuda()
print(args)

##  增加模块优化器(5个/***或考虑1-2个做联合(jointly)优化，后期考虑参数共享部分)和实验超参设置
press_optimizer = optim.Adam(press_MSM.parameters(), lr=args.learning_rate, weight_decay=params_SAS['weight_decay'])
##
scheduler_press = StepLR(press_optimizer, step_size=params_SAS['lr_decay_epoch'], gamma=params_SAS['lr_decay'])
## Transform
utilPath = osp.join(
    args.basePath,
    args.dataset_name,
    params_SAS['dataset_util'])
#
mean_st = np.load(osp.join(utilPath, 'mean_st_modal.npy'))
std_st = np.load(osp.join(utilPath, 'std_st_modal.npy'))
transformer_4D = transforms.Compose([
    transforms.Normalize(mean=mean_st, std=std_st)
    ])
## Dataset
modalOrd = modalOrdinalDataset(args,
    params_SAS['preset_iter'] * params_SAS['train_batch_size'] * len(args.device_ids),
    params_SAS['startDate'],
    params_SAS['endDate'],
    dataset_path,
    params_SAS['split'],
    prepRange,
    params_SAS['resample_range'],
    transformer_4D
    )
OrdDataLoader = data.DataLoader(
    modalOrd, 
#     batch_size=params_SAS['train_batch_size'],
    batch_size=64,
    drop_last=True,
    shuffle=True,
    pin_memory=True,
    num_workers=args.num_workers,
    )
#
ECSequence = modalOrd.seq_length
ECSize = modalOrd.currScale
ECChannel = modalOrd.channel_nums
#
modalGD_F = modalGribDataset(args, 
    params_SAS['preset_iter'] * params_SAS['test_batch_size'] * len(args.device_ids),
    params_SAS['resample_range'],
    params_SAS['startDate'],
    params_SAS['endDate'],
    dataset_path,
    params_SAS['split'],
    transformer_4D,
    False
    )
modalGD_FDataLoader = data.DataLoader(
    modalGD_F, 
#     batch_size=params_SAS['test_batch_size'],
    batch_size=16,
    drop_last=True,
    shuffle=False,
    pin_memory=True,
    num_workers=args.num_workers,
    )

## 多尺度范围
scale_range = np.arange(3, ECSize+params_SAS['multiScale_step'], params_SAS['multiScale_step'])
EC_center = floorHalf(ECSize)
## epoch前设置较大的best_loss
p_bestLoss = 10000

"""
压强target有问题(讲道理华东区域压强不会差太多），欠拟合，跳变非常的大，需要jupyter上面验证一下，先保存模型
"""
print ('Pre-Training press_MSM...')
for epoch in range(args.preEpoch):
    print ("current_epoch:", epoch)
    for batch_idx, (data, seqPrep, _, seqTem, seqPress, seqWind, seqDew) in enumerate(OrdDataLoader):
        print ('update TrainbatchSize-{}...'.format(batch_idx))
        ##  N * C * D * H * W  → update  N1D * C * H * W   N1D / N1D * 2
        data_press, pressTarget = updateMeBatch4Label(data, seqPress, args.dropRate,'press')
        
        print ('pressTarget:', pressTarget)
        
        for ci, curScale in tqdm.tqdm(
            enumerate(scale_range), total = len(scale_range),
            desc = 'Training MSM', ncols = 80,
            leave=False):
            print ('current scale-{}:'.format(curScale))
            ## Crop current scale
            curScale_press = cropOneScale4EC(data_press, curScale, EC_center)
            ## Training
            press_MSM.train()
            #
            press_optimizer.zero_grad()
            #
            press_outputs = press_MSM(curScale_press)
            # BP
            BP_ME(press_outputs, pressTarget, press_optimizer, 'press')

   
    ## Inference
    if epoch%params_SAS['training_inv']==0:
        p_losses = []
        for batch_idx, (data, seqPrep, _, seqTem, seqPress, seqWind, seqDew) in enumerate(modalGD_FDataLoader):
            print ('update testBatchSize-{}...'.format(batch_idx))
            ##  N * C * D * H * W  → update  N1D * C * H * W   N1D / N1D * 2 
            data_press, pressTarget = updateMeBatch4Label(data, seqPress, args.dropRate,'press')
            pressTarget = pressTarget.unsqueeze(-1)
#             print ('pressTarget:', pressTarget)
                          
            for ci, curScale in tqdm.tqdm(
                    enumerate(scale_range), total = len(scale_range),
                    desc = 'Inferencing MSM', ncols = 80,
                    leave=False):
                print ('current scale-{}:'.format(curScale))
                press_MSM.eval()
                ## Crop current scale
                curScale_press = cropOneScale4EC(data_press, curScale, EC_center)
                ## Inference
                with torch.no_grad():
                    p_outputs = press_MSM(curScale_press)
                    
                    #
                    p_loss = F.mse_loss(p_outputs, pressTarget).item()
                    #
                    p_losses.append(p_loss)
    
        # 计算当前epoch中的所有minibatch平均loss的均值
        p_loss_avg = np.mean(p_losses)
        print('epoch-{}-Test-pressLoss-{}'.format(epoch + 1, p_loss_avg))

        
        #
#         if epoch>=3 and p_bestLoss > p_loss_avg:
#             p_bestLoss = p_loss_avg
        press_weights = press_MSM.state_dict()
        print('saving-epoch-{}-Test-pressLoss-{}'.format(epoch + 1, p_bestLoss))
        #
        torch.save(press_weights, osp.join(SMS_save_path, 'press', 'pretrained-MSM-press.pth'))            
                  
    ##
    scheduler_press.step()      