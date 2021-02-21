from torch.utils import data
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import tqdm
from datasets.TASDataset import modalGribDataset, modalOrdinalDataset
from utils.util_preTraining import updateMeBatch4Label, flagMeBatch4Label, generateProbs4t1Label, load_preTraining_MEs, interFlagIndice
# from aging.dataset import data_prefetcher
import os.path as osp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from models.TAS import Encoder, CNN_3D, OrdinalRegressionModel, rainFallClassification
from apex import amp
from apex.parallel import DistributedDataParallel
from utils.util_main import calOptimTS4Sample, reduce_tensor, featuresNorm, save_osConfig, generateOneHot, generateMeanNoisy, floorHalf 
from utils.util_evals_I import rainFocalLoss, evalAcc_rain_non_rain, ordinalLoss, eval_TS, get_TS_indice, eval_errors, calTSLoss 
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import sys
import os

## equivalent to ablate to SFM
class TAS(object):
    def __init__(self,
        args,
        dataset_type,
        dataset_util,
        preTrain_MTM,
        #         
        multiLength_step,
        startDate,
        endDate,
        uniform_t1sLen,
        #
        set_min_CtemValue, 
        set_max_CtemValue,
        set_Ctem_space,
        set_min_CpressValue,
        set_max_CpressValue,
        set_Cpress_space,
        set_max_CwindVoc,
        set_max_CwindDre,
        set_CwVoc_space,
        set_CwDre_space,
        set_min_CdewValue, 
        set_max_CdewValue,
        set_Cdew_space,
        set_max_CrainValue,
        set_Crain_space,                  
        #         
        in_channel,
        ts_out_channel,          
        ts_seq,
        alpha,
        beta,
        init_lr,
        lr_decay,
        lr_decay_epoch,
        lr_decay_iter,
        init_best_error,
        weight_decay,
        training_inv,
        origin_ecSize,
        uniform_upDownFeaSize, 
        ed_out_channel,         
        train_batch_size,
        test_batch_size,                 
        resample_range,
        split,
        noiseStd,
        set_ord_prep_space,
        set_ord_max_prepValue,
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
        self.MTM_save_path = osp.join(args.basePath, preTrain_MTM)        
        #
        self.dataset_util = dataset_util
        self.multiLength_step = multiLength_step
        self.out_path = args.save_root
        self.startDate = startDate
        self.endDate = endDate
        self.uniform_t1sLen = uniform_t1sLen
        #
        self.set_min_CtemValue = set_min_CtemValue
        self.set_max_CtemValue = set_max_CtemValue
        self.set_Ctem_space = set_Ctem_space
        self.set_min_CpressValue = set_min_CpressValue
        self.set_max_CpressValue = set_max_CpressValue
        self.et_Cpress_space = set_Cpress_space
        self.set_max_CwindVoc = set_max_CwindVoc
        self.set_max_CwindDre = set_max_CwindDre
        self.set_CwVoc_space = set_CwVoc_space
        self.set_CwDre_space = set_CwDre_space
        self.set_min_CdewValue = set_min_CdewValue
        self.set_max_CdewValue = set_max_CdewValue
        self.set_Cdew_space = set_Cdew_space
        self.set_max_CrainValue = set_max_CrainValue
        self.set_Crain_space  = set_Crain_space   
        #
        self.in_channel = in_channel
        self.ts_out_channel = ts_out_channel
        self.ed_out_channel = ed_out_channel
        self.ts_seq = ts_seq
        self.alpha = alpha
        self.beta = beta
        self.init_lr = init_lr
        self.lr_decay = lr_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_iter = lr_decay_iter
        self.init_best_error = init_best_error
        self.weight_decay = weight_decay
        self.training_inv = training_inv
        self.origin_ecSize = origin_ecSize
        self.uniform_upDownFeaSize = uniform_upDownFeaSize
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size        
        self.resample_range = resample_range
        self.split = split
        self.noiseStd = noiseStd
        ##
        self.spaces = set_ord_prep_space
        self.prep_ord_Range = np.arange(self.spaces, set_ord_max_prepValue,self.spaces)
        self.nClass = self.prep_ord_Range.shape[0]        
        self.CtemRange = np.arange(set_min_CtemValue,set_max_CtemValue+set_Ctem_space, set_Ctem_space)
        self.CpressRange = np.arange(set_min_CpressValue,set_max_CpressValue+set_Cpress_space, set_Cpress_space)
        self.CwindVocRange = np.arange(0,set_max_CwindVoc+set_CwVoc_space, set_CwVoc_space) 
        self.CwindDreRange = np.arange(0,set_max_CwindDre+set_CwDre_space, set_CwDre_space)
        self.CdewRange = np.arange(set_min_CdewValue,set_max_CdewValue+set_Cdew_space, set_Cdew_space)                                   
        self.CrainRange = np.arange(0,set_max_CrainValue+set_Crain_space, set_Crain_space)     
        ##
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
        #
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.apex_save_dir, exist_ok=True)
        os.makedirs(self.hyperParams_save_dir, exist_ok=True)        
        # apex env
        dist.init_process_group(backend='nccl',
                                init_method='env://')
        
                
        # MTM
        from pre_training.MTM import C3AE_3DI, C3AE_3DII
        tem_MTM = C3AE_3DI(self.ts_seq, self.CtemRange, self.uniform_upDownFeaSize,  self.ed_out_channel).cuda()
        press_MTM = C3AE_3DI(self.ts_seq, self.CpressRange, self.uniform_upDownFeaSize,  self.ed_out_channel).cuda()
        wind_MTM = C3AE_3DII(self.ts_seq, self.CwindDreRange, self.CwindVocRange, self.uniform_upDownFeaSize, self.ed_out_channel).cuda()
        dew_MTM = C3AE_3DI(self.ts_seq, self.CdewRange, self.uniform_upDownFeaSize, self.ed_out_channel).cuda()
        rain_MTM =C3AE_3DI(self.ts_seq, self.CrainRange, self.uniform_upDownFeaSize, self.ed_out_channel).cuda()
        # load_preTraining model
        tem_MTM = amp.initialize(tem_MTM, opt_level="O0", verbosity=0)
        press_MTM = amp.initialize(press_MTM, opt_level="O0", verbosity=0)
        wind_MTM = amp.initialize(wind_MTM, opt_level="O0", verbosity=0)
        dew_MTM = amp.initialize(dew_MTM, opt_level="O0", verbosity=0)
        rain_MTM = amp.initialize(rain_MTM, opt_level="O0", verbosity=0)
       # load_preTraining model 
        tem_MTM, press_MTM, wind_MTM, dew_MTM, rain_MTM = load_preTraining_MEs(self.MTM_save_path, tem_MTM, press_MTM, wind_MTM, dew_MTM, rain_MTM, 'MTM')      
        # parallel and eval
        self.tem_MTM = DistributedDataParallel(tem_MTM, delay_allreduce=True)
        self.press_MTM = DistributedDataParallel(press_MTM, delay_allreduce=True)
        self.wind_MTM = DistributedDataParallel(wind_MTM, delay_allreduce=True)
        self.dew_MTM = DistributedDataParallel(dew_MTM, delay_allreduce=True)
        self.rain_MTM = DistributedDataParallel(rain_MTM, delay_allreduce=True)
        #                               
        self.tem_MTM.eval()
        self.press_MTM.eval()
        self.wind_MTM.eval()
        self.dew_MTM.eval()
        self.rain_MTM.eval()                                                                   
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
            self.prep_ord_Range,
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
        self.ECSequence = modalOrd.seq_length                                       
        # 1-5
        self.ts_range = np.arange(1, self.ECSequence, self.multiLength_step)
        self.EC_center = floorHalf(self.ECSize)
        self.ECChannel = modalOrd.channel_nums 
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
        En = Encoder(self.uniform_upDownFeaSize, self.in_channel, self.ed_out_channel).cuda()
        # TS Module
        CNN3D = CNN_3D(self.uniform_upDownFeaSize, self.ed_out_channel, self.ts_out_channel).cuda() 
        # OR
        regressionModel = OrdinalRegressionModel(self.nClass).cuda()
        # Classifier
        rainFallClassifierModel = rainFallClassification().cuda()
        ## Optim init
        self.regressionOptim = torch.optim.Adam([
            {'params': En.parameters(), 'lr': self.init_lr,
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
        self.TAS_modules, self.regressionOptim = amp.initialize([En, CNN3D, regressionModel], self.regressionOptim, opt_level='O0', verbosity=1)
        self.rainFallClassifierModel, self.rainFallOptim = amp.initialize(rainFallClassifierModel, self.rainFallOptim, opt_level='O0', verbosity=1)    
        self.TAS_modules[0] = DistributedDataParallel(self.TAS_modules[0], delay_allreduce=True)
        self.TAS_modules[1] = DistributedDataParallel(self.TAS_modules[1], delay_allreduce =True)                                
        self.TAS_modules[2] = DistributedDataParallel(self.TAS_modules[2], delay_allreduce =True)              
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
            self.train_TAS()               
            self.scheduler_Reg.step()
            self.scheduler_Rf.step()
            self.epoch_iterations = self.epoch_iterations + 1
                                                                                       
    ######## MTM ###########
    def MTM(self, encoderTS, b_t1_Prep, b_t1_Tem, b_t1_Press, b_t1_Wind, b_t1_Dew):
        # can ignore → secondary screening
        #  generate:N1*range / N1*range*2  
        b_t1_Tem, temProbs_, N2D_tem_labelFlags = generateProbs4t1Label(b_t1_Tem, self.CtemRange, 'tem')
        b_t1_Press, pressProbs_, N2D_press_labelFlags = generateProbs4t1Label(b_t1_Press, self.CpressRange, 'press')
        b_t1_Wind, windProbs_, N2D_wind_labelFlags = generateProbs4t1Label(b_t1_Wind,(self.CwindDreRange, self.CwindVocRange), 'wind')
        b_t1_Dew, dewProbs_, N2D_dew_labelFlags = generateProbs4t1Label(b_t1_Dew, self.CdewRange, 'dew')
        b_t1_Prep, rainProbs_, N2D_rain_labelFlags = generateProbs4t1Label(b_t1_Prep, self.CrainRange, 'rain') 
        #
        N2D_labelFlags = interFlagIndice(N2D_tem_labelFlags, N2D_press_labelFlags, N2D_wind_labelFlags, N2D_dew_labelFlags, N2D_rain_labelFlags)
        '''
        ????结果还是有问题，label为什么二次筛选又变少了????
        '''
#         print ('available ts-feature for miniBatch:', N2D_labelFlags.count(5)) 
#         print ('N2D_labelFlags:', N2D_labelFlags)
        # best length on curr miniBach
        bestLength4TS = []
        # update TS features 
        updateTS_N2D_features = []
        ## 对于每个站点的所有时间尺度下来找最优尺度并保存
        for idx, tsf_stat in enumerate(encoderTS):
            ts_losses_list = []
            ts_f_list = []
            # 截取和保存统一时间尺度
            if N2D_labelFlags[idx]!=5:
                uniTs_stat = tsf_stat[:,(self.ts_seq-self.uniform_t1sLen-1):,:,:].unsqueeze(0)
                updateTS_N2D_features.append(uniTs_stat)                
                bestLength4TS.append(self.uniform_t1sLen)
                print ('station error!!!')            
                continue
            #
            for ti, curTs in tqdm.tqdm(
                enumerate(self.ts_range), 
                total=len(self.ts_range),
                desc='selecting adaptive ts for D-CNN TS Features', ncols=80,
                leave=False):
                with torch.no_grad():
                    curTs_f = tsf_stat[:,(self.ts_seq-curTs-1):,:,:].unsqueeze(0)
                    # 
                    t_probPreds, t_valuePreds = self.tem_MTM(curTs_f)
                    p_probPreds, p_valuePreds = self.press_MTM(curTs_f)
                    # 
                    prob_wdPreds, value_wdPreds, prob_wsPreds, value_wsPreds = self.wind_MTM(curTs_f)
                    d_probPreds, d_valuePreds = self.dew_MTM(curTs_f)
                    r_probPreds, r_valuePreds = self.rain_MTM(curTs_f)
                    #### 集成eval 判定1
                    t_loss  = calTSLoss(t_valuePreds, b_t1_Tem[idx].unsqueeze(0), t_probPreds, temProbs_[idx].unsqueeze(0), 'tem')
                    p_loss  = calTSLoss(p_valuePreds, b_t1_Press[idx].unsqueeze(0), p_probPreds, pressProbs_[idx].unsqueeze(0), 'press')                   
                    w_loss  = calTSLoss((value_wdPreds, value_wsPreds), b_t1_Wind[idx], (prob_wdPreds, prob_wsPreds, idx), windProbs_, 'wind')                   
                    d_loss  = calTSLoss(d_valuePreds, b_t1_Dew[idx].unsqueeze(0), d_probPreds, dewProbs_[idx].unsqueeze(0), 'dew')                   
                    r_loss  = calTSLoss(r_valuePreds, b_t1_Prep[idx].unsqueeze(0), r_probPreds, rainProbs_[idx].unsqueeze(0), 'rain')                   
                    #
                    tpwdr_loss  = 0.2 * t_loss + 0.2 * p_loss + 0.2 * w_loss + 0.2 * d_loss + 0.2 * r_loss
                                                              
                    ts_f_list.append(curTs_f)
                    ts_losses_list.append(tpwdr_loss)
                    
            # 计算当前样本的最优尺度，截取并返回当前站点最优尺度及对应的EC
            optim_ts, optim_tsf = calOptimTS4Sample(ts_losses_list, ts_f_list, self.ts_range)
            # N*(D-1)个不同尺度的站点EC
            updateTS_N2D_features.append(optim_tsf)
            bestLength4TS.append(optim_ts)
        return updateTS_N2D_features, bestLength4TS 
    
                     
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
#             self.writer.add_scalars('Loss_{}'.format("RainFall-Classifier"), {
#                 'cls_loss': cls_loss},
#                 self.epoch_iterations)            

#             self.writer.add_scalars('Accuracy_{}'.format("RainFall-Classifier"), {
#                 'acc_non_rain': accuracy[0],
#                 'acc_rain': accuracy[1],                   
#                 'acc_total': accuracy[2]}, 
#                 self.epoch_iterations) 
            
    ########Train TAS###########            
    def train_TAS(self):
        self.TAS_modules[0].train()
        self.TAS_modules[1].train()
        self.TAS_modules[2].train()
        
        for batch_idx, (data, seqPrep, seq_ordLabels, seqTem, seqPress, seqWind, seqDew) in tqdm.tqdm(enumerate(self.modalOrdDataLoader), 
            total=len(self.modalOrdDataLoader), 
            desc='Train TAS epoch=%d' % self.epoch, 
            ncols=100, 
            leave=False):
            self.regressionOptim.zero_grad()
            data = data.to(device=self.device)
            ## Encoder
            # N*C*D*H*W→ Cuda
            cpu_stats = self.TAS_modules[0](data, self.train_batch_size, True)
            ND_stats = torch.from_numpy(cpu_stats.astype(np.float32)).cuda()   
            # MTM:  select temporal scale
            ts_D2_features_l, ts_D2_length_l  = self.MTM(ND_stats, seqPrep[:,-2], seqTem[:,-2], seqPress[:,-2], seqWind[:,-2], seqDew[:,-2])
            # 3DCNN                           
            out_3dFeatures = self.TAS_modules[1](ts_D2_features_l, True)
            # OR
            ordinalPreds = self.TAS_modules[2](out_3dFeatures)                                       # 
            ordLabels = seq_ordLabels.to(self.device)[:,-1]
            ## Forward
            ## Loss
            ce, emd = self.ordinalLoss(ordinalPreds, ordLabels)
            loss = self.alpha * ce + self.beta * emd
            with amp.scale_loss(loss, self.regressionOptim) as scaled_loss:
                scaled_loss.backward()
            self.regressionOptim.step()
            ord_loss = reduce_tensor(ce.detach(), self.num_devices).item()
            # visual
#             self.writer.add_scalars('Loss_{}'.format("TAS"), {
#                 'ord_loss': ord_loss
#                 }, 
#                 self.epoch_iterations)      
        print ('epoch-{}-last_miniBatch-ordLoss:'.format(self.epoch), ord_loss)

    ########Evaluation TAS###########            
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
        self.TAS_modules[0].eval()
        self.TAS_modules[1].eval()
        self.TAS_modules[2].eval()
        self.rainFallClassifierModel.eval()
                        
        for batch_idx, (data, seqPrep, seqRainClass, seqTem, seqPress, seqWind, seqDew) in tqdm.tqdm(
            enumerate(self.modalGD_FDataLoader), 
            total = len(self.modalGD_FDataLoader),
            desc = 'Test TAS:', ncols = 80,
            leave=False):
            # Data non-ablation
            # N*D
            rainNumpy = seqRainClass.numpy()[:,-1]
            gt_micaps = seqPrep.numpy()[-1]
            data = data.to(device=self.device)
            target = seqPrep.to(device=self.device)[:,-1].unsqueeze(1)

            # Inference
            with torch.no_grad():
                cpu_stats = self.TAS_modules[0](data, self.test_batch_size, False)           
                ND_stats = torch.from_numpy(cpu_stats.astype(np.float32)).cuda()
                # select temporal scale
                ts_D2_features_l, ts_D2_length_l  = self.MTM(ND_stats, seqPrep[:,-2], seqTem[:,-2], seqPress[:,-2], seqWind[:,-2], seqDew[:,-2])
                # N * C * H * W                  
                out_3dFeatures = self.TAS_modules[1](ts_D2_features_l, False)
                # ORD                        
                predictValues = self.TAS_modules[2](out_3dFeatures)
                #           
                rainPreds = self.rainFallClassifierModel(data[:,:,-1,:,:])               
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
#                 self.writer.add_scalars('test_{}_{}'.format("TAS","iteration"), {
#                     'reg_Loss': reg_loss}, 
#                     self.epoch_iterations)
                                
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
#         self.writer.add_scalars('test_{}_{}'.format("TAS","epoch"), {
#             'TS_0.1': ts[0],
#             'TS_1': ts[1],
#             'TS_10': ts[2],
#             'ACC_non_Rain': rain_acc1,
#             'ACC_Rain': rain_acc2,
#             'ACC_Rain_cls': rain_acc3}, 
#             self.epoch)        

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
            info["EncoderParam"] = self.TAS_modules[0].state_dict()
            info["CNN3DParam"] = self.TAS_modules[1].state_dict()                       
            info["ORParam"] = self.TAS_modules[2].state_dict()            
            info["rainFallParam"] = self.rainFallClassifierModel.state_dict()
            info["optimParam"] = self.regressionOptim.state_dict()
            info["amp"] = amp.state_dict()            
            
#             torch.save(info, 
#                 os.path.join(self.models_save_dir, str(self.epoch) + "_TAS_checkpoint.pth")
#                 )
            torch.save(info, 
                os.path.join(self.models_save_dir, 'fixed' + "_TAS_checkpoint.pth")
                )
