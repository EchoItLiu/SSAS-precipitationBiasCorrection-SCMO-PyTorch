from torch.utils import data
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import tqdm
from datasets.SASDataset import modalGribDataset, modalOrdinalDataset
from utils.util_preTraining import updateMeBatch4Label, flagMeBatch4Label, load_preTraining_MEs, interFlagIndice
# from aging.dataset import data_prefetcher
import os.path as osp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from models.SAS import uniEncoder, ED_DCNN, CNN_3D, OrdinalRegressionModel, rainFallClassification
from apex import amp
from apex.parallel import DistributedDataParallel
from utils.util_main import calOptimScale4Sample, cropOneScale4Sample, reduce_tensor, featuresNorm, save_osConfig, generateOneHot, generateMeanNoisy, floorHalf 
from utils.util_evals_I import rainFocalLoss, evalAcc_rain_non_rain, ordinalLoss, eval_TS, get_TS_indice, eval_errors 
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import sys
import os

## equivalent to ablate to STM
class SAS(object):
    def __init__(self,
                 args,
                 dataset_type,
                 dataset_util,
                 preTrain_MSM,
                 multiScale_step, 
                 startDate,
                 endDate,
                 uniform_ecSize,
                 uniform_upDownFeaSize,
                 uniform_t1sLen,
                 in_channel,
                 ed_out_channel,
                 ts_out_channel,
                 ts_seq,
                 ####                 
                 alpha,
                 beta,
                 init_lr,
                 lr_decay,
                 lr_decay_epoch,
                 lr_decay_iter,
                 init_best_error,
                 weight_decay,
                 training_inv,
                 train_batch_size,
                 test_batch_size,                 
                 resample_range,
                 split,
                 noiseStd,
                 set_prep_space,
                 set_max_prepValue,
                 max_epoch,
                 preset_iter,
                 max_iter,
                 save_iter,
                 **kwargs
                 ):
        self.args = args
        self.dataset_name = args.dataset_name
        self.dataset_type = dataset_type
        self.dataset_path = osp.join(args.basePath, args.dataset_name, dataset_type)
        self.SMS_save_path = osp.join(args.basePath, preTrain_MSM)        
        self.dataset_util = dataset_util
        self.multiScale_step = multiScale_step
        self.out_path = args.save_root
        self.startDate = startDate
        self.endDate = endDate
        self.uniform_ecSize = uniform_ecSize 
        self.uni_upDownFSize = uniform_upDownFeaSize
        self.in_channel = in_channel
        self.ed_out_channel = ed_out_channel
        self.ts_out_channel = ts_out_channel
        self.ts_seq = ts_seq
        ##
        self.alpha = alpha
        self.beta = beta
        self.init_lr = init_lr
        self.lr_decay = lr_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_iter = lr_decay_iter
        self.init_best_error = init_best_error
        self.weight_decay = weight_decay
        self.training_inv = training_inv
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size        
        self.resample_range = resample_range
        self.split = split
        self.noiseStd = noiseStd
        self.spaces = set_prep_space
        self.prepRange = np.arange(self.spaces, set_max_prepValue,self.spaces)
        self.nClass = self.prepRange.shape[0]
        self.max_epoch = max_epoch
        self.preset_iter = preset_iter
        self.max_iter = max_iter
        self.save_iter = save_iter
        ##
        self.local_rank = self.args.local_rank
        self.device_ids = args.device_ids 
        self.num_devices = len(self.device_ids)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = args.num_workers

                     
    def __repr__(self):
        return self.__class__.__name__

    def init_module(self):
        self.models_save_dir = osp.join(self.args.save_root, 'save_models')
        self.apex_save_dir = osp.join(self.args.save_root, 'save_apex')
        self.hyperParams_save_dir = osp.join(self.args.save_root, 'save_hyper')
        
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.apex_save_dir, exist_ok=True)
        os.makedirs(self.hyperParams_save_dir, exist_ok=True)        
        # apex env
        dist.init_process_group(backend='nccl',
                                init_method='env://')
                
        # 加入预训练模型
        from pre_training.MSM import DeformConvNetI, DeformConvNetII
        tem_MSM = DeformConvNetI(self.in_channel).cuda()
        press_MSM = DeformConvNetI(self.in_channel).cuda()
        wind_MSM = DeformConvNetII(self.in_channel).cuda()
        dew_MSM = DeformConvNetI(self.in_channel).cuda()
        rain_MSM =DeformConvNetI(self.in_channel).cuda()
        # load_preTraining model
        tem_MSM = amp.initialize(tem_MSM, opt_level="O0", verbosity=0)
        press_MSM = amp.initialize(press_MSM, opt_level="O0", verbosity=0)
        wind_MSM = amp.initialize(wind_MSM, opt_level="O0", verbosity=0)
        dew_MSM = amp.initialize(dew_MSM, opt_level="O0", verbosity=0)
        rain_MSM = amp.initialize(rain_MSM, opt_level="O0", verbosity=0)
       # load_preTraining model 
        tem_MSM, press_MSM, wind_MSM, dew_MSM, rain_MSM = load_preTraining_MEs(self.SMS_save_path, tem_MSM, press_MSM, wind_MSM, dew_MSM, rain_MSM, 'MSM')   
#         print ('----load successfully----')
        
        # parallel and eval
        self.tem_MSM = DistributedDataParallel(tem_MSM, delay_allreduce=True)
        self.press_MSM = DistributedDataParallel(press_MSM, delay_allreduce=True)
        self.wind_MSM = DistributedDataParallel(wind_MSM, delay_allreduce=True)
        self.dew_MSM = DistributedDataParallel(dew_MSM, delay_allreduce=True)
        self.rain_MSM = DistributedDataParallel(rain_MSM, delay_allreduce=True)
#         print ('----Distributed successfully----')      
        self.tem_MSM.eval()
        self.press_MSM.eval()
        self.wind_MSM.eval()
        self.dew_MSM.eval()
        self.rain_MSM.eval()
        # visual
        self.writer = None
        self.writer = SummaryWriter(osp.join(self.args.save_root, 'tensorboard'))         
        # EC-Pre pipeline
        featuresMean_st, featuresStd_st = featuresNorm(self.args, self.dataset_name, self.dataset_util, 'modal_ST')
        featuresMean_s, featuresStd_s = featuresNorm(self.args, self.dataset_name, self.dataset_util, 'modal_S')           
        ## 4DTransformer
        transformer_4D = transforms.Compose([
        transforms.Normalize(mean=featuresMean_st, std=featuresStd_st)
    ])        
        ## 3DTransformer
        transformer_3D = transforms.Compose([
        transforms.Normalize(mean=featuresMean_s, std=featuresStd_s)
    ]) 
        ## Prefetchers
        self.prefetchers = []
        pass
        ## DataLoader
        modalOrd = modalOrdinalDataset(self.args,
            self.preset_iter * self.train_batch_size * self.num_devices,
            self.startDate,
            self.endDate,
            self.dataset_path,
            self.split,
            self.prepRange,
            self.resample_range,
            transformer_4D
            )
        ordTrainSet_sampler = data.distributed.DistributedSampler(modalOrd)
        self.modalOrdDataLoader = data.DataLoader(
            modalOrd, 
            batch_size=self.train_batch_size,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=ordTrainSet_sampler
            )            
        #
        self.ECSize = modalOrd.currScale
        self.scale_range = np.arange(3, self.ECSize + self.multiScale_step, self.multiScale_step)
        self.EC_center = floorHalf(self.ECSize)
        self.ECChannel = modalOrd.channel_nums 
        self.ECSequence = modalOrd.seq_length
        #
        modalGD_T = modalGribDataset(self.args, 
            self.preset_iter * self.train_batch_size * self.num_devices,
            self.resample_range,
            self.startDate,
            self.endDate,
            self.dataset_path,
            self.split,
            transformer_3D,
            True
            )
        rainFallDataset_sampler = data.distributed.DistributedSampler(modalGD_T)        
        self.modalGD_TDataLoader = data.DataLoader(
            modalGD_T, 
            batch_size=self.train_batch_size,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=rainFallDataset_sampler
            )        
        # Test Dataset
        modalGD_F = modalGribDataset(self.args, 
            self.preset_iter * self.test_batch_size * self.num_devices,
            self.resample_range,
            self.startDate,
            self.endDate,
            self.dataset_path,
            self.split,
            transformer_4D,
            False
            )
        TestSet_sampler = data.distributed.DistributedSampler(modalGD_F, shuffle=False) 
        self.modalGD_FDataLoader = data.DataLoader(
            modalGD_F, 
            batch_size=self.test_batch_size,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=TestSet_sampler
            )             
        
    ## 初始化Pipeline模型及优化器             
    def init_model(self): 
        # Encoder
        uniE = uniEncoder(self.uni_upDownFSize, self.in_channel).cuda()
        EDCNN = ED_DCNN(self.ed_out_channel).cuda()
        # TS Module
        CNN3D = CNN_3D(self.uni_upDownFSize, self.ts_out_channel).cuda()
        # OR
        regressionModel = OrdinalRegressionModel(self.nClass).cuda()
        # Classifier
        rainFallClassifierModel = rainFallClassification().cuda()
        ## Optim init
        self.regressionOptim = torch.optim.Adam([
            {'params': uniE.parameters(), 'lr': self.init_lr,
             'weight_decay': self.weight_decay},
            {'params': EDCNN.parameters(), 'lr': self.init_lr,
             'weight_decay': self.weight_decay},
            {'params': CNN3D.parameters(), 'lr': self.init_lr,
             'weight_decay': self.weight_decay},            
            {'params': regressionModel.parameters(), 'lr': self.init_lr * 10,
             'weight_decay': self.weight_decay * 10},            
            ],
            lr=self.init_lr * 10, 
            weight_decay=self.weight_decay * 10
            )
    
        self.rainFallOptim = torch.optim.Adam(
            rainFallClassifierModel.parameters(), 
            lr=self.init_lr * 10
            ) 
        # Apex
        self.SAS_modules, self.regressionOptim = amp.initialize([uniE, EDCNN,CNN3D, regressionModel], self.regressionOptim, opt_level='O0', verbosity=1)
        self.rainFallClassifierModel, self.rainFallOptim = amp.initialize(rainFallClassifierModel, self.rainFallOptim, opt_level='O0', verbosity=1)    
        self.SAS_modules[0] = DistributedDataParallel(self.SAS_modules[0], delay_allreduce=True)
        self.SAS_modules[1] = DistributedDataParallel(self.SAS_modules[1], delay_allreduce =True)
        self.SAS_modules[2] = DistributedDataParallel(self.SAS_modules[2], delay_allreduce =True)        
        self.SAS_modules[3] = DistributedDataParallel(self.SAS_modules[3], delay_allreduce =True)        
        
        # lr decay
        self.scheduler_Reg = StepLR(self.regressionOptim, step_size=self.lr_decay_epoch, gamma=self.lr_decay)
        self.scheduler_Rf = StepLR(self.rainFallOptim, step_size=self.lr_decay_epoch, gamma=self.lr_decay)
        # user-defined losses
        self.rainFocalLoss = rainFocalLoss(2, alpha = 0.25, gamma = 2).cuda() 
        self.ordinalLoss = ordinalLoss(self.spaces).cuda()        
        # save os config 
        save_osConfig(self.args, self.hyperParams_save_dir)
                    
    def fit(self):
        self.init_model()
        # if max_epoch
        self.epoch_iterations = 0         
        ## fit
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            # binary classifier
            self.train_bin_cls()
            if self.max_epoch % self.training_inv == 0:
                # inference
                self.bias_location_prep()

            # spatial adaptive selection
            self.train_SAS()               
            self.scheduler_Reg.step()
            self.scheduler_Rf.step()
            self.epoch_iterations = self.epoch_iterations + 1
            
    ######## MSM ###########
    def MSM(self, miniBatch, bSeqPrep, bSeqTem, bSeqPress, bSeqWind, bSeqDew):
        #  N * C * (D-1) * H * W  → update  N(D-1) * C * H * W   N(D-1) / N(D-1) * 2
        ND1_date_, temTarget_, ND1_tem_labelFlags = flagMeBatch4Label(miniBatch[:,:,:-1,:,:], bSeqTem[:,:-1], 'tem')
        _, pressTarget_, ND1_press_labelFlags = flagMeBatch4Label(miniBatch[:,:,:-1,:,:], bSeqPress[:,:-1], 'press')
        _, windTarget_, ND1_wind_labelFlags = flagMeBatch4Label(miniBatch[:,:,:-1,:,:], bSeqWind[:,:-1], 'wind')
        _, dewTarget_, ND1_dew_labelFlags = flagMeBatch4Label(miniBatch[:,:,:-1,:,:], bSeqDew[:,:-1], 'dew')
        _, prepTarget_, ND1_rain_labelFlags = flagMeBatch4Label(miniBatch[:,:,:-1,:,:], bSeqPrep[:,:-1], 'rain') 
        
        ND1_labelFlags = interFlagIndice(ND1_tem_labelFlags, ND1_press_labelFlags, ND1_wind_labelFlags, ND1_dew_labelFlags, ND1_rain_labelFlags)
        print ('available adP-EC for miniBatch:', ND1_labelFlags.count(5))
        # best scales on curr miniBach
        bestScales4Station = []
        # update scales 
        updateScales_ND1_data = []
        ## 对于每个站点的所有尺度下来找最优尺度并保存
        for idx, stat in enumerate(ND1_date_):
            scale_losses_list = []
            scale_stases_list = []
            # 截取和保存统一尺度
            if ND1_labelFlags[idx]!=5:
                uniScale_stat = calOptimScale4Sample(scale_losses_list, scale_stases_list, self.scale_range, stat, self.uniform_ecSize, self.EC_center, 'uni')
                updateScales_ND1_data.append(uniScale_stat)                
                bestScales4Station.append(self.uniform_ecSize)
                continue
            #
            for ci, curScale in tqdm.tqdm(
                enumerate(self.scale_range), total = len(self.scale_range),
                desc = 'selecting adaptive scale for previous ECs', ncols = 80,
                leave=False):
                with torch.no_grad():
                    curScale_stat = cropOneScale4Sample(stat, curScale, self.EC_center)
                    #
                    t_output = self.tem_MSM(curScale_stat)
                    p_output = self.press_MSM(curScale_stat)
                    # 2
                    w_output = self.wind_MSM(curScale_stat)
                    d_output = self.dew_MSM(curScale_stat)
                    r_output = self.rain_MSM(curScale_stat)
                    #
                    t_loss = F.mse_loss(t_output.squeeze(0), temTarget_[idx].unsqueeze(0)).item()
                    p_loss = F.mse_loss(p_output.squeeze(0), pressTarget_[idx].unsqueeze(0)).item()
                    w_loss = (0.6 * F.mse_loss(w_output[0][0], windTarget_[idx][0]) + 0.4*F.mse_loss(w_output[0][1], windTarget_[idx][1])).item()
                    d_loss = F.mse_loss(d_output.squeeze(0), dewTarget_[idx].unsqueeze(0)).item()
                    r_loss = F.mse_loss(r_output.squeeze(0), prepTarget_[idx].unsqueeze(0)).item()
                    #
                    tpwdr_loss  = 0.2 * t_loss + 0.2 * p_loss + 0.2 * w_loss + 0.2 * d_loss + 0.2 * r_loss
                    scale_stases_list.append(curScale_stat)
                    scale_losses_list.append(tpwdr_loss)
                    
            # 计算当前样本的最优尺度，截取并返回当前站点最优尺度及对应的EC
            opimScale_stat, opimScale = calOptimScale4Sample(scale_losses_list, scale_stases_list, self.scale_range, stat, None, None, 'optim')
            # N*(D-1)个不同尺度的站点EC
            updateScales_ND1_data.append(opimScale_stat)
            bestScales4Station.append(opimScale)
            
        return updateScales_ND1_data, bestScales4Station 
            

                  
    ########Train Classifier###########     
    def train_bin_cls(self):
        self.rainFallClassifierModel.train()
        for batch_idx, (data, target, rainClass) in tqdm.tqdm(
            enumerate(self.modalGD_TDataLoader), 
            total=len(self.modalGD_TDataLoader), 
            desc='Train RainFall Classification epoch=%d' % self.epoch, 
            ncols=100, 
            leave=False):
            
            ## Data
            data = data.to(device=self.device)
            logitNumpy = rainClass.numpy()
            logits = rainClass.to(device=self.device)
           
            self.rainFallOptim.zero_grad()
            preds = self.rainFallClassifierModel(data)
            classificationLoss = self.rainFocalLoss(preds, logits)
            
            with amp.scale_loss(classificationLoss, self.rainFallOptim) as scaled_loss:
                scaled_loss.backward() 
            self.rainFallOptim.step()
            # all Reduce
            cls_loss = reduce_tensor(classificationLoss.detach(), self.num_devices).item()
            
            # evaluation 
            predsSoftmax = F.softmax(preds, dim = 1)
            predicted = torch.argmax(predsSoftmax, dim = 1).cpu().numpy()
            accuracy = evalAcc_rain_non_rain(predicted, logitNumpy)
            print("\nTrain Rain Fall Classification Accuracy : ", accuracy)
            
            # visual
            self.writer.add_scalars('Loss_{}'.format("RainFall-Classifier"), {
                'cls_loss': cls_loss},
                self.epoch_iterations)            

            self.writer.add_scalars('Accuracy_{}'.format("RainFall-Classifier"), {
                'acc_non_rain': accuracy[0],
                'acc_rain': accuracy[1],                   
                'acc_total': accuracy[2]}, 
                self.epoch_iterations) 
            
    ########Train SAS###########            
    def train_SAS(self):
        self.SAS_modules[0].train()
        self.SAS_modules[1].train()
        self.SAS_modules[2].train()
        self.SAS_modules[3].train()        
        for batch_idx, (data, seqPrep, seq_ordLabels, seqTem, seqPress, seqWind, seqDew) in tqdm.tqdm(enumerate(self.modalOrdDataLoader), 
            total=len(self.modalOrdDataLoader), 
            desc='Train SAS epoch=%d' % self.epoch, 
            ncols=100, 
            leave=False):
            self.regressionOptim.zero_grad()
            data = data.to(device=self.device)
            
            ## MSM - selecting adaptive scale automatically
            flattenScales_ND1_data, flattenScales_ND1_Stat  = self.MSM(data, seqPrep, seqTem, seqPress, seqWind, seqDew)
            ## Encoder
            # Uni:N*C*(D-1)*H*W_NDarray → Cuda
            cpu_uni_stats = self.SAS_modules[0](flattenScales_ND1_data, self.train_batch_size, 'D-1', True)
            ND1_uni_stats = torch.from_numpy(cpu_uni_stats.astype(np.float32)).cuda()   
            # Uni:N*C*1*H*W
            N_uni_stats = self.SAS_modules[0](data[:,:,-1,:,:], self.train_batch_size, 'D-th', True)
            N_uni_stats = torch.unsqueeze(N_uni_stats, 2)
            # concat: N*C*D*H*W
            ND_uni_stats = torch.cat((ND1_uni_stats, N_uni_stats), 2)
            # D-CNN: N*C*D*H*W
            cpu_encoder_ND_stats = self.SAS_modules[1](ND_uni_stats, self.train_batch_size, True)
            encoder_ND_stats = torch.from_numpy(cpu_encoder_ND_stats.astype(np.float32)).cuda()
            # N * C * H * W
            tsEncoder_stats = self.SAS_modules[2](encoder_ND_stats, True)
            ordLabels = seq_ordLabels.to(self.device)[:,-1]
            ## Forward
            ordinalPreds = self.SAS_modules[3](tsEncoder_stats)
            ## Loss
            ce, emd = self.ordinalLoss(ordinalPreds, ordLabels)
            loss = self.alpha * ce + self.beta * emd
            with amp.scale_loss(loss, self.regressionOptim) as scaled_loss:
                scaled_loss.backward()
            self.regressionOptim.step()
            ord_loss = reduce_tensor(ce.detach(), self.num_devices).item()
#             self.writer.add_scalars('Loss_{}'.format("SAS"), {
#                 'ord_loss': ord_loss
#                 }, 
#                 self.epoch_iterations)      
        print ('epoch-{}-last_miniBatch-ordLoss:'.format(self.epoch), ord_loss)

    ########Evaluation SAS###########            
    def bias_location_prep(self):
        tsthreas = [0.1, 1, 10]
        rain_acc = []
        totalRegressionLoss = []
        totalReconstructLoss = []
        tp = [0]*len(tsthreas)
        fp = [0]*len(tsthreas)
        fn = [0]*len(tsthreas)
        total_error = 0
        p_error = 0
        ps_error = 0
        p_count = 0
        total_error = 0
        total_count = 0
        self.SAS_modules[0].eval()
        self.SAS_modules[1].eval()
        self.SAS_modules[2].eval()
        self.SAS_modules[3].eval()          
        self.rainFallClassifierModel.eval()
                        
        for batch_idx, (data, seqPrep, seqRainClass, seqTem, seqPress, seqWind, seqDew) in tqdm.tqdm(
            enumerate(self.modalGD_FDataLoader), 
            total = len(self.modalGD_FDataLoader),
            desc = 'Test SAS:', ncols = 80,
            leave=False):
            print ('---')
            # Data non-ablation
            # N*D
            rainNumpy = seqRainClass.numpy()[:,-1]
            gt_micaps = seqPrep.numpy()[-1]
            data = data.to(device=self.device)
            target = seqPrep.to(device=self.device)[:,-1].unsqueeze(1)
            # 
            flattenScales_ND1_data, flattenScales_ND1_Stat  = self.MSM(data, seqPrep, seqTem, seqPress, seqWind, seqDew)
            # Inference
            with torch.no_grad():
                cpu_uni_stats = self.SAS_modules[0](flattenScales_ND1_data, self.test_batch_size, 'D-1', False)                
                ND1_uni_stats = torch.from_numpy(cpu_uni_stats.astype(np.float32)).cuda() 
                N_uni_stats = self.SAS_modules[0](data[:,:,-1,:,:], self.test_batch_size, 'D-th', False)
                N_uni_stats = torch.unsqueeze(N_uni_stats, 2)
                ND_uni_stats = torch.cat((ND1_uni_stats, N_uni_stats), 2)
                cpu_encoder_ND_stats = self.SAS_modules[1](ND_uni_stats, self.test_batch_size, False)
                encoder_ND_stats = torch.from_numpy(cpu_encoder_ND_stats.astype(np.float32)).cuda()
                print ('encoder_ND_stats_size:', encoder_ND_stats.size())                
                #
                tsEncoder_stats = self.SAS_modules[2](encoder_ND_stats, True)
                print ('tsEncoder_stats_size:', tsEncoder_stats.size())
                predictValues = self.SAS_modules[3](tsEncoder_stats)
                rainPreds = self.rainFallClassifierModel(data[:,:,-1,:,:])               
                #
                rainPredsSoftMax = F.softmax(rainPreds, dim = 1)
                rainOneHotMask = generateOneHot(rainPredsSoftMax).to(self.device)
                regressionValues = self.spaces * (torch.sum((predictValues>0.5).float(), dim=1).view(-1, 1))
                zeros = torch.zeros(regressionValues.size()).to(self.device)
                regressionValues = torch.matmul(rainOneHotMask, 
                    torch.cat([zeros,
                    regressionValues], 
                    dim=1).unsqueeze(-1)).squeeze(-1)
                predictNumpy = regressionValues.cpu().numpy()                
                # Loss
                regressionLoss = F.mse_loss(regressionValues, target)
                totalRegressionLoss.append(regressionLoss.item())
                reg_loss = reduce_tensor(regressionLoss.detach(), self.num_devices).item()
                # Evaluating params 
                rainPredicted = torch.argmax(rainPredsSoftMax, dim = 1).cpu().numpy()                    
                rain_acc.append(evalAcc_rain_non_rain(rainPredicted, rainNumpy))
                gapValues = np.abs(predictNumpy - gt_micaps)
                
                total_error += np.sum(gapValues)
                p_ae = (gt_micaps > 0.1) * gapValues
                p_error += np.sum(p_ae)
                ps_error += np.sum(p_ae ** 2)
                p_count += np.sum(gt_micaps > 0.1)
                total_count += gapValues.shape[0]               
                tp_cur, fp_cur, fn_cur = get_TS_indice(tsthreas, gt_micaps, predictNumpy)
                tp = [e+tp_cur[i] for i,e in enumerate(tp)]
                fp = [e+fp_cur[i] for i,e in enumerate(fp)]
                fn = [e+fn_cur[i] for i,e in enumerate(fn)]
                
                # visual
                self.writer.add_scalars('test_{}_{}'.format("SAS","iteration"), {
                    'reg_Loss': reg_loss}, 
                    self.epoch_iterations)
                                
        # Evaluations
        ts = eval_TS(tsthreas, tp, fp, fn)
        totalAverageError, pAverageError, psAverageError = eval_errors(total_error, p_error, ps_error, p_count, total_count)
        totalLoss = np.mean(totalRegressionLoss)
        totalRLoss = np.mean(totalReconstructLoss)
        tsDisplay = list(zip(tp, fp, fn, ts))
        rain_acc1 = round(np.mean([rain_acc[i][0] for i in range(len(rain_acc))]), 5)
        rain_acc2 = round(np.mean([rain_acc[i][1] for i in range(len(rain_acc))]), 5)
        rain_acc3 = round(np.mean([rain_acc[i][2] for i in range(len(rain_acc))]), 5)
        
        # visual
        self.writer.add_scalars('test_{}_{}'.format("SAS","epoch"), {
            'TS_0.1': ts[0],
            'TS_1': ts[1],
            'TS_10': ts[2],
            'ACC_non_Rain': rain_acc1,
            'ACC_Rain': rain_acc2,
            'ACC_Rain_cls': rain_acc3}, 
            self.epoch)        

        info = {"test_regression_loss": totalLoss,
                "test_reconstruct_loss": totalRLoss,
                "aver_gap": totalAverageError,
                "aver_p_gap": pAverageError,
                "aver_ps_gap": psAverageError,
                "p_num": p_count,
                "ts_score": tsDisplay,
                "test_rain_classification_accuracy": [rain_acc1, rain_acc2, rain_acc3],
                }
        print("======= Epoch {} Test Result Show =======".format(self.epoch + 1))
        print(info)
        
        if totalAverageError < self.init_best_error:
            self.best_error = totalAverageError
            self.best_res_epoch = self.epoch
            info["epoch"] = self.epoch
            info["uniEncoderParam"] = self.SAS_modules[0].state_dict()
            info["D-CNNParam"] = self.SAS_modules[1].state_dict()
            info["ORParam"] = self.SAS_modules[2].state_dict()            
            info["rainFallParam"] = self.rainFallClassifierModel.state_dict()
            info["optimParam"] = self.regressionOptim.state_dict()
            info["amp"] = amp.state_dict()            
            
            torch.save(info, 
                os.path.join(self.models_save_dir, str(self.epoch) + "_SAS_checkpoint.pth")
                )
