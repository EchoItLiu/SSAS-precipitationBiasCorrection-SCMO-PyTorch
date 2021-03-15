import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

"""
 xxx    
"""
##Rain Focal Loss
klLoss = nn.KLDivLoss()

class rainFocalLoss(nn.Module):
    def __init__(self, 
        nClass, alpha=0.75, 
        gamma=2, 
        balance_index=-1, 
        size_average=True):
        super(rainFocalLoss, self).__init__()
        
        # nClass = 2
        self.nClass = nClass
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.size_average = size_average

        if self.alpha is None:
            # [2×1]
            self.alpha = torch.ones(self.nClass, 1)
            
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.nClass
            # [1×1]
            self.alpha = torch.FloatTensor(alpha).unsqueeze(-1)
            self.alpha = self.alpha / self.alpha.sum()
            
        elif isinstance(self.alpha, float):
            # [1,1]
            alpha = torch.ones(self.nClass, 1)
            # [0.25, 0.25]
            alpha = alpha * (1 - self.alpha)
            # [0.25, 0.25]
            alpha[balance_index] = self.alpha
            self.alpha = alpha
            
        else:
            raise NotImplementedError

    # 
    def forward(self, input, target):
        # get dimensionality (N × 2)
        if input.dim() > 2:
            # N × 1 × 2  
            input = input.view(input.size(0), input.size(1), -1)
        # N × 1
        target = target.view(-1, 1)
        epsilon = 1e-10
        alpha = self.alpha
        
        # assign CUDA:X to alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        # re-scale the elements in last dim lie in the range [0, 1] and sum to 1
        # softmax again 
        logit = F.softmax(input, dim = -1)
        
        # binary 0/1 [N × 1]
        idx = target.cpu().long()
        
        # make zero for Tensor[size(N, 2)]
        one_hot_key = torch.FloatTensor(target.size(0), self.nClass).zero_()
        
        # insert 1 to the first dim of one_hot_key according to idx
        # one-hot binary (0,1)/(1,0) Tensor[size(N, 2)]
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        
        # same CUDA: x
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
        
        # the possibility for specific classifier(only save GT-classifier) [N × 1]
        pt = (one_hot_key * logit).sum(1) + epsilon
        # log pt
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        
        # Cal Rain Focal Loss for binary classifier ( make gradient close to 0 ) [N × 1]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        
        # Mean/Sum
        if self.size_average:
            #  [1]
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
class ordinalLoss(nn.Module):
    def __init__(self, space):
        super(ordinalLoss, self).__init__()
        
        """------- term added to improve numerical stability -------"""
        self.epilson = 1e-10
        self.gamma = 2
        self.space = space

    def forward(self, x, y):
        batchSize = x.size(0)
        gamma = self.gamma
        
        ##  Mean Focal Loss with CE
        ce = - (torch.sum(
            torch.pow((1 - x), gamma) * torch.log(x + self.epilson).mul(y) +
            torch.pow(x, gamma) * torch.log(1 - x + self.epilson).mul(1 - y))
        ) / batchSize

        # cal histogram distribution for all x 
        x_soft = F.softmax(x, dim = 1)
        # cal histogram distribution for all y
        y_soft = F.softmax(y, dim = 1)
        ## EMD (Earth Mover's Distance) Loss 
        x_cumsum = torch.cumsum(x_soft, dim = 1)
        y_cumsum = torch.cumsum(y_soft, dim = 1)
        
        EMD_loss = torch.sum((x_cumsum - y_cumsum) ** 2)
        return ce, EMD_loss
    
def evalAcc_rain_non_rain(pred, rainLabel):
    classCorrect = [0] * 2
    classCount = [0] * 2
    accuracy = [0] * 3        
     
    for i in range(2):
        classCorrect[i] = np.sum((pred == i) * (rainLabel == i))
        classCount[i] = np.sum(rainLabel == i)           
        
    for i in range(2):
        accuracy[i] = round(classCorrect[i] / classCount[i], 5)
    accuracy[2] = round(sum(classCorrect) / sum(classCount), 5)    
    return accuracy

def eval_TS(tsthreas, TPs, FPs, FNs):
    TS = [0]*len(tsthreas)  
    for i in range(len(tsthreas)):
        TS[i] = round(TPs[i] / (TPs[i] + FPs[i] + FNs[i]), 5)
    return TS

def get_TS_indice(tsthreas, gtMicaps, predNumpy):
    
    tp = [0]*len(tsthreas)
    fp = [0]*len(tsthreas)
    fn = [0]*len(tsthreas)
    # 11mm/3mm tsthreas: 0.1
    # 0.1mm/3mm/11mm tsthreas: 1, 0.1mm可以使得tp=0,fn=0,fp=1 0/0+0+1可能去贡献了分母(也是预测错误)
    # 0.1mm/3mm/11mm tsthreas: 10,  
    
    for i, threas in enumerate(tsthreas):           
        tp[i] = np.sum((gtMicaps>=threas) * (predNumpy>=threas))
        fp[i] = np.sum((gtMicaps<threas) * (predNumpy>=threas))
        fn[i] = np.sum((gtMicaps>=threas) * (predNumpy<threas))          
    return tp, fp, fn


def eval_errors(totalErrors, pErrors, psErrors, pCounts, totalCounts):
    totalAverageError = round(totalErrors / totalCounts, 5)
    pAverageError = round(pErrors / pCounts, 5)
    psAverageError = round(psErrors / pCounts - pAverageError ** 2, 5)
    return totalAverageError, pAverageError, psAverageError


def calTSLoss(valuePreds, b_t1_Me, probPreds, Probs_Me, flag):
    if flag=='wind':       
        loss0_mse = F.mse_loss(valuePreds[0], b_t1_Me[0]).item()
        loss1_mse = F.mse_loss(valuePreds[1], b_t1_Me[1]).item()
        loss_mse = 0.6*loss0_mse + 0.4*loss1_mse
        #
        b_idx = probPreds[2]    
        loss0_kl = klLoss(probPreds[0].squeeze(0), Probs_Me[0][b_idx]).item()
        loss1_kl = klLoss(probPreds[1].squeeze(0), Probs_Me[1][b_idx]).item()
        loss_kl = 0.6*loss0_kl + 0.4*loss1_kl
        #
        ts_loss = 0.5*loss_mse + 0.5*loss_kl        

    else:      
        mse = F.mse_loss(valuePreds, b_t1_Me).item()                  
        kl = klLoss(probPreds, Probs_Me).item()                    
        #
        ts_loss = 0.5 * mse + 0.5 * kl
        
    return ts_loss
