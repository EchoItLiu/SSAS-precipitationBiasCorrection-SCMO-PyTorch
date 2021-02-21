import os
from torch.utils import data
import torch
import torch.nn as nn
import numpy as np
# import torchvision
from torchvision import transforms
import tqdm
import sys
from datasets.OBADataset import ordinalGribDataset, gribDataset
# from aging.dataset import data_prefetcher
import os.path as osp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from models.OBA import OrdinalRegressionModel, rainFallClassification, AutoencoderBN
from apex import amp
from apex.parallel import DistributedDataParallel
from utils.util_main import reduce_tensor, featuresNorm, save_osConfig, generateOneHot, generateMeanNoisy 
from utils.util_evals_I import rainFocalLoss, evalAcc_rain_non_rain, ordinalLoss, eval_TS, get_TS_indice, eval_errors 
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

class OBA(object):
    def __init__(self,
                 args,
                 dataset_type,
                 dataset_util,
                 startDate,
                 endDate,
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
        self.dataset_util = dataset_util
        self.out_path = args.save_root
        self.startDate = startDate
        self.endDate = endDate
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
        self.num_devices = len(self.args.device_ids)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = args.num_workers
                       
        ## pretraining module save path
        pass

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
     
        self.writer = None
        self.writer = SummaryWriter(osp.join(self.args.save_root, 'tensorboard'))
       
        # EC-Pre pipeline
        featuresMean, featuresStd = featuresNorm(self.args, self.dataset_name, self.dataset_util, 'OBA')
        transformer = transforms.Compose([
        transforms.Normalize(mean=featuresMean, std=featuresStd)
    ])
        
        ##    
        regressionDataSet = ordinalGribDataset(self.args,
            self.preset_iter * self.train_batch_size * self.num_devices,
            self.startDate,
            self.endDate,
            self.dataset_path,
            self.split,
            self.prepRange,
            self.resample_range,
            transform = transformer
            )    

        regressionDataSet_sampler = data.distributed.DistributedSampler(regressionDataSet)
        self.ordTrainLoader = data.DataLoader(
            regressionDataSet, 
            batch_size=self.train_batch_size,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=regressionDataSet_sampler
            )    
            # Prefetcher
#         self.prefetchers = []  
            
#         self.prefetchers.append(data_prefetcher(train_loader, lambda x: (normlize(x[0]), x[1], normlize(x[2]))))
        rainFallDataset = gribDataset(self.args, 
            self.preset_iter * self.train_batch_size * self.num_devices,
            self.resample_range,
            self.startDate,
            self.endDate,
            self.dataset_path,
            self.split,                          
            transform = transformer,
            isTrain = True
            )
    
        rainFallDataset_sampler = data.distributed.DistributedSampler(rainFallDataset)
        self.trainRainFallLoader = data.DataLoader(
            rainFallDataset, 
            batch_size=self.train_batch_size,
            drop_last = True,
            shuffle = False,
            pin_memory=True,
            num_workers=self.num_workers,        
            sampler=rainFallDataset_sampler      
            )        

        testDataSet = gribDataset(self.args, 
            self.max_iter * self.test_batch_size * self.num_devices,
            self.resample_range,
            self.startDate,
            self.endDate,
            self.dataset_path,
            self.split,                                                                
            transform = transformer, 
            isTrain = False
            )
        
        testDataSet_sampler = data.distributed.DistributedSampler(testDataSet)
        self.testDataLoader = data.DataLoader(testDataSet, 
            batch_size=self.test_batch_size, 
            drop_last=True,        
            pin_memory=True, 
            num_workers=self.num_workers, 
            sampler=testDataSet_sampler
            )
         
    def init_model(self):         
        autoModel = AutoencoderBN().cuda()
        regressionModel = OrdinalRegressionModel(self.nClass).cuda()
        rainFallClassifierModel = rainFallClassification().cuda()
        
        ## Optim init
        self.regressionOptim = torch.optim.Adam([
            {'params': regressionModel.parameters(), 'lr': self.init_lr * 10,
             'weight_decay': self.weight_decay * 10},
            {'params': autoModel.parameters(), 'lr': self.init_lr,
             'weight_decay': self.weight_decay},],
            lr=self.init_lr * 10, 
            weight_decay=self.weight_decay * 10
            )
    
        self.rainFallOptim = torch.optim.Adam(
            rainFallClassifierModel.parameters(), 
            lr=self.init_lr * 10
            ) 
        
        # Apex
        self.OBA_modules, self.regressionOptim = amp.initialize([autoModel, regressionModel], self.regressionOptim, opt_level='O0', verbosity=1)
        self.rainFallClassifierModel, self.rainFallOptim = amp.initialize(rainFallClassifierModel, self.rainFallOptim, opt_level='O0', verbosity=1)    
        self.OBA_modules[0] = DistributedDataParallel(self.OBA_modules[0], delay_allreduce=True)
        self.OBA_modules[1] = DistributedDataParallel(self.OBA_modules[1], delay_allreduce =True)
        
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
            # ordinal regression
            self.train_ord_reg()               
            self.scheduler_Reg.step()
            self.scheduler_Rf.step()
            self.epoch_iterations = self.epoch_iterations + 1
            
    ########Train Classifier###########     
    def train_bin_cls(self):
        self.rainFallClassifierModel.train()
        for batch_idx, (data, target, rainClass, rainMask) in tqdm.tqdm(
            enumerate(self.trainRainFallLoader), 
            total=len(self.trainRainFallLoader), 
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
            
    ########Train OBA###########            
    def train_ord_reg(self):
        self.OBA_modules[0].train()
        self.OBA_modules[1].train()
        for batch_idx, (data, target, ordinalLabels) in tqdm.tqdm(
            enumerate(self.ordTrainLoader), 
            total=len(self.ordTrainLoader), 
            desc='Train Regression epoch=%d' % self.epoch, 
            ncols=100, 
            leave=False):
            
            ## Data
            noiseMean = generateMeanNoisy(data.size(0), data.size(1), None, data.size(2), '3D')
            noise = torch.normal(mean=noiseMean, std=self.noiseStd)
            noisedData = data + noise
            data = data.to(self.device)
            noisedData = noisedData.to(self.device)
            ordinalLabels = ordinalLabels.to(self.device)
            ## Forward
            self.regressionOptim.zero_grad()
            encoder, decoder = self.OBA_modules[0](noisedData)
            ordinalPreds = self.OBA_modules[1](encoder)
            ## Loss                      
            constructLoss = F.mse_loss(decoder, data)
            ce, emd = self.ordinalLoss(ordinalPreds, ordinalLabels)
            loss = constructLoss + self.alpha * ce + self.beta * emd
            with amp.scale_loss(loss, self.regressionOptim) as scaled_loss:
                scaled_loss.backward()
            self.regressionOptim.step()
            struct_loss = reduce_tensor(constructLoss.detach(), self.num_devices).item()
            ord_loss = reduce_tensor(ce.detach(), self.num_devices).item()
            
            self.writer.add_scalars('Loss_{}'.format("OBA"), {
                'ord_loss': ord_loss,
                'struct_loss': struct_loss}, 
                self.epoch_iterations)      
        print ('epoch-{}-last_miniBatch-ordLoss:'.format(self.epoch), ord_loss)
            
            
    ########Evaluation OBA###########            
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
        self.OBA_modules[0].eval()
        self.OBA_modules[1].eval()
        self.rainFallClassifierModel.eval()
        
        for batch_idx, (data, target, rainClass, rainMask) in tqdm.tqdm(
            enumerate(self.testDataLoader), total = len(self.testDataLoader),
            desc = 'Test Test Data :', ncols = 80,
            leave=False): 
            
            ## Data
            rainNumpy = rainClass.numpy()
            gt_micaps = target.numpy()
            data = data.to(device=self.device)
            target = target.to(device=self.device)
            
            # Inference
            with torch.no_grad():
                encoder, decoder = self.OBA_modules[0](data)
                predictValues = self.OBA_modules[1](encoder)
                rainPreds = self.rainFallClassifierModel(data)
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
#                 print ('regressionValues:', regressionValues)
#                 print ('target:', target)
                regressionLoss = F.mse_loss(regressionValues, target)
                reconstructLoss = F.mse_loss(decoder, data)
                totalRegressionLoss.append(regressionLoss.item())
                totalReconstructLoss.append(reconstructLoss.item())
                reg_loss = reduce_tensor(regressionLoss.detach(), self.num_devices).item()
                rec_loss = reduce_tensor(reconstructLoss.detach(), self.num_devices).item()
                
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
                self.writer.add_scalars('test_{}_{}'.format("OBA","iteration"), {
#                     'pred': predictNumpy,
#                     'obs': gt_micaps,
                    'reg_Loss': reg_loss,
                    'rec_Loss': rec_loss}, 
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
        self.writer.add_scalars('test_{}_{}'.format("OBA","epoch"), {
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
            info["modelParam"] = self.OBA_modules[0].state_dict()
            info["regressionParam"] = self.OBA_modules[1].state_dict()
            info["rainFallParam"] = self.rainFallClassifierModel.state_dict()
            info["optimParam"] = self.regressionOptim.state_dict()
            info["amp"] = amp.state_dict()            
            
            torch.save(info, 
                os.path.join(self.models_save_dir, str(self.epoch) + "_OBA_checkpoint.pth")
                )

        