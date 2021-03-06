import torch
import numpy as np
import yaml
import sys
sys.path.append("/home/yqliu/EC-BiasCorrecting")
sys.path.append("/home/yqliu/EC-BiasCorrecting/pre_training")
# print (sys.path)
import os
import os.path as osp
from torch.utils import data
import torchvision
from datasets.STASDataset import modalOrdinalDataset, modalGribDataset 
from utils.util_preTraining import updateTSMeBatch4Label, BP_ME_TS
from utils.util_main import cropOneLength4EC, get_file_name
import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch import nn
import copy
import argparse
from torchvision import transforms

'''
python pre_training/MTM_pretraining/W_MTM_Pre.py -c ./config/SHO.yaml -r 0.3 -d 8 9
'''
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', help='config file', type=str, required=True)
parser.add_argument('--dropRate', '-r', help='drop some samples randomly', type=float, required=True)
parser.add_argument('--device_ids', '-d', nargs='*', help='devices id to be used', type=int)
parser.add_argument('--learning_rate', help='devices id to be used', type=float, default=1e-4)
parser.add_argument("--num_workers", help='num_workers', default=16, type=int)
parser.add_argument('--in_channel', help='meteorological factor nums', type=int, default=57)
parser.add_argument('--in_seq', help='meteorological temporal scale', type=int, default=6)
parser.add_argument("--basePath", default = "/mnt/pami14/yqliu/EC-BiasCorrecting")
# parser.add_argument("--preEpoch", help='pretraining epoch', type=int, default=50)
parser.add_argument("--preEpoch", help='pretraining epoch', type=int, default=10000)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## params参数导出
with open(args.config) as f:
    config = yaml.load(f)
args.dataset_name = get_file_name(args.config)
## dic
params_STAS = config['STAS']
#
dataset_path = osp.join(args.basePath, args.dataset_name, params_STAS['dataset_type'])
MTM_save_path = osp.join(args.basePath, params_STAS['preTrain_MTM'])

#
prepRange = np.arange(params_STAS['set_ord_prep_space'], params_STAS['set_ord_max_prepValue'], params_STAS['set_ord_prep_space'])
dwindRange = np.arange(0,params_STAS['set_max_CwindDre']+params_STAS['set_CwDre_space'], params_STAS['set_CwDre_space'])
swindRange = np.arange(0,params_STAS['set_max_CwindVoc']+params_STAS['set_CwVoc_space'], params_STAS['set_CwVoc_space'])
windRange = (dwindRange, swindRange)
nClass = prepRange.shape[0]

## 选择MSM
def selectMTM(modelType):
    if modelType == 1:
        from pre_training.MTM import C3AE_3DI, middleFeaturesCNN  
        model1 = C3AE_3DI
        model2 = middleFeaturesCNN
    elif modelType == 2:
        from pre_training.MTM import C3AE_3DII, middleFeaturesCNN
        model1 = C3AE_3DII
        model2 = middleFeaturesCNN        
    elif modelType == 3:
        from pre_training.MTM import C3AE_3DIII, middleFeaturesCNN
        model1 = C3AE_3DIII
        model2 = middleFeaturesCNN     
    else:
        raise NotImplementedError       
    return model2, model1

def initMTM(element_flag):
    if element_flag=='tem':
        midCNN, C3AE = selectMTM(1)
        init1 = midCNN(args.in_channel, params_STAS['uniform_upDownFeaSize'], params_STAS['ed_out_channel'])
        init2 = C3AE(args.in_seq, temRange, params_STAS['uniform_upDownFeaSize'], params_STAS['ed_out_channel'])
    if element_flag=='press':
        midCNN, C3AE = selectMTM(1)
        init1 = midCNN(args.in_channel, params_STAS['uniform_upDownFeaSize'], params_STAS['ed_out_channel'])
        init2 = C3AE(args.in_seq, pressRange, params_STAS['uniform_upDownFeaSize'], params_STAS['ed_out_channel'])
    if element_flag=='wind':
        midCNN, C3AE = selectMTM(2)
        init1 = midCNN(args.in_channel, params_STAS['uniform_upDownFeaSize'], params_STAS['ed_out_channel'])
        init2 = C3AE(args.in_seq, dwindRange, swindRange, params_STAS['uniform_upDownFeaSize'], params_STAS['ed_out_channel'])
    if element_flag=='dew':
        midCNN, C3AE = selectMTM(1)
        init1 = midCNN(args.in_channel, params_STAS['uniform_upDownFeaSize'], params_STAS['ed_out_channel'])
        init2 = C3AE(args.in_seq, dewRange, params_STAS['uniform_upDownFeaSize'], params_STAS['ed_out_channel'])
    if element_flag=='rain':
        midCNN, C3AE = selectMTM(1)
        init1 = midCNN(args.in_channel, params_STAS['uniform_upDownFeaSize'], params_STAS['ed_out_channel'])
        init2 = C3AE(args.in_seq, rainRange, params_STAS['uniform_upDownFeaSize'], params_STAS['ed_out_channel'])
    return init1, init2
        
## 5个预训练模型需要的子模块
middleFCN, C3AE_3D = initMTM('wind')
# nnParallel
torch.cuda.set_device(args.device_ids[0])
if len(args.device_ids) > 1:
    middleFCN = nn.DataParallel(middleFCN, device_ids=args.device_ids).cuda()
    C3AE_3D = nn.DataParallel(C3AE_3D, device_ids=args.device_ids).cuda()
    klLoss = nn.KLDivLoss()
else:
    middleFCN = middleFCN.cuda()
    C3AE_3D = C3AE_3D.cuda()
    # Loss
    klLoss = nn.KLDivLoss()
    print(args)

## 增加模块优化器(5个/***或考虑1-2个做联合(jointly)优化，后期考虑参数共享部分)和实验超参设置
'''
？？？一定要强调MSM和MTM的温压风湿降水都是联合优化,最好做联合优化，可以考虑联合fine-tuning？？？
'''
wind_optimizer = torch.optim.Adam([
    {'params': middleFCN.parameters(), 'lr': args.learning_rate * 10,
     'weight_decay': params_STAS['weight_decay'] * 10},
    {'params': C3AE_3D.parameters(), 'lr': args.learning_rate,
     'weight_decay': params_STAS['weight_decay']}
    ],
    lr=args.learning_rate * 10, 
    weight_decay=params_STAS['weight_decay'] * 10
    )
##
scheduler_wind = StepLR(wind_optimizer, step_size=params_STAS['lr_decay_epoch'], gamma=params_STAS['lr_decay'])
## Transform
utilPath = osp.join(
    args.basePath,
    args.dataset_name,
    params_STAS['dataset_util'])
#
mean_st = np.load(osp.join(utilPath, 'mean_st_modal.npy'))
std_st = np.load(osp.join(utilPath, 'std_st_modal.npy'))
transformer_4D = transforms.Compose([
    transforms.Normalize(mean=mean_st, std=std_st)
    ])
## Dataset
# Trainset
modalOrd = modalOrdinalDataset(args,
    params_STAS['preset_iter'] * params_STAS['train_batch_size'] * len(args.device_ids),
    params_STAS['startDate'],
    params_STAS['endDate'],
    dataset_path,
    params_STAS['split'],
    prepRange,
    params_STAS['resample_range'],
    transformer_4D
    )
OrdDataLoader = data.DataLoader(
    modalOrd, 
    batch_size=params_STAS['train_batch_size'],
    drop_last=True,
    shuffle=True,
    pin_memory=True,
    num_workers=args.num_workers,
    )
#
ECSequence = modalOrd.seq_length
ECSize = modalOrd.currScale
ECChannel = modalOrd.channel_nums
# Testset
modalGD_F = modalGribDataset(args, 
    params_STAS['preset_iter'] * params_STAS['test_batch_size'] * len(args.device_ids),
    params_STAS['resample_range'],
    params_STAS['startDate'],
    params_STAS['endDate'],
    dataset_path,
    params_STAS['split'],
    transformer_4D,
    False
    )
modalGD_FDataLoader = data.DataLoader(
    modalGD_F, 
    batch_size=params_STAS['test_batch_size'],
    drop_last=True,
    shuffle=False,
    pin_memory=True,
    num_workers=args.num_workers,
    )

## 时间序列范围 t-1,t-2,..0
length_range = np.arange(0, ECSequence-params_STAS['multiLength_step'], params_STAS['multiLength_step'])[::-1]
## epoch前设置较大的best_loss
w_bestLoss = 10000

print ('Pre-Training wind_MTM...')
for epoch in range(args.preEpoch):
    print ("current_epoch:", epoch)
    for batch_idx, (data, seqPrep, _, seqTem, seqPress, seqWind, seqDew) in enumerate(OrdDataLoader):
        print ('update TrainbatchSize-{}...'.format(batch_idx))
        ##  N * C * D * H * W  → update N1 * D * C * H * W  N1*range / N1*range*2  N1*2 N1 / N1*2        
        data_wind, windTarget, windProbs = updateTSMeBatch4Label(data, seqWind[:,-1], windRange, args.dropRate, 'wind')
        
        for ci, curLength in tqdm.tqdm(
            enumerate(length_range), total = len(length_range),
            desc = 'Training MTM', ncols = 80,
            leave=False):
            print ('current -{}:'.format(curLength))
            ## Crop current scale
            curLength_wind = cropOneLength4EC(data_wind, curLength)
            ## Training
            middleFCN.train()
            C3AE_3D.train()
            
            #
            wind_optimizer.zero_grad()
            # N1*C1*D*H_u*W_u
            midF = middleFCN(curLength_wind, True)
            prob_wdPreds, value_wdPreds, prob_wsPreds, value_wsPreds = C3AE_3D(midF)
            
            probPreds = (prob_wdPreds, prob_wsPreds)
            valuePreds = (value_wdPreds, value_wsPreds)
            
            # BP
            BP_ME_TS(valuePreds, probPreds, windTarget, windProbs, wind_optimizer, 'wind')

    ## Inference
    if epoch%params_STAS['training_inv']==0:
        w_losses = []
        for batch_idx, (data, seqPrep, _, seqTem, seqPress, seqWind, seqDew) in enumerate(modalGD_FDataLoader):
            print ('update testBatchSize-{}...'.format(batch_idx))
            ##  N * C * D * H * W  → update  N1 * D * C * H * W  N1*range / N1*range*2  N1*2 N1 / N1*2
            data_wind, windTarget, windProbs = updateTSMeBatch4Label(data, seqWind[:,-1], windRange, args.dropRate, 'wind')
            temTarget = windTarget.unsqueeze(-1)
#             temProbs = temProbs.unsqueeze(-1)

#             print ('ori_temTarget_size:', temTarget.size())
#             print ('ori_data_tem_size:', data_tem.size())
                      
            for ci, curLength in tqdm.tqdm(
                    enumerate(length_range), total = len(length_range),
                    desc = 'Inferencing MTM', ncols = 80,
                    leave=False):
                print ('current length-{}:'.format(curLength))
                middleFCN.eval()
                C3AE_3D.eval()
                ## Crop current scale
                curLength_wind = cropOneLength4EC(data_wind, curLength)
                ## Inference
                with torch.no_grad():
                    midF = middleFCN(curLength_wind, False)
                    prob_wdPreds, value_wdPreds, prob_wsPreds, value_wsPreds = C3AE_3D(midF) 
                    #
                    loss0_mse = F.mse_loss(value_wdPreds, windTarget[:,0]).item()
                    loss1_mse = F.mse_loss(value_wsPreds, windTarget[:,1]).item()
                    loss_mse = 0.6*loss0_mse + 0.4*loss1_mse
                    #
                    loss0_kl = klLoss(prob_wdPreds, windProbs[0]).item()
                    loss1_kl = klLoss(prob_wsPreds, windProbs[1]).item()
                    loss_kl = 0.6*loss0_kl + 0.4*loss1_kl
                    #
                    loss_total = 0.5*loss_mse + 0.5*loss_kl
                    #
                    w_losses.append(loss_total)
        # 计算当前epoch中的所有minibatch平均loss的均值
        w_loss_avg = np.mean(w_losses)   
        print('epoch-{}-Test-windLoss-{}'.format(epoch + 1, w_loss_avg))
        ## save 5 ME modules  copy.deepcopy(model.state_dict())
        if epoch>=3 and w_bestLoss > w_loss_avg:
            w_bestLoss = w_loss_avg
            wind_weights = C3AE_3D.state_dict()
            print('saving-epoch-{}-Test-windLoss-{}'.format(epoch + 1, w_bestLoss))
            # spatial module for temperature and override
            torch.save(wind_weights, osp.join(MTM_save_path, 'wind', 'pretrained-MTM-wind.pth'))
            
    ##
    scheduler_wind.step()
