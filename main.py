import argparse
import torch
import os.path as osp
import yaml

torch.multiprocessing.set_sharing_strategy('file_system')

def get_model(args):
    model_name = args.model_name
    if model_name == 'STAS':
        from experiments.STAS_naive import STAS
        model_type = STAS
    elif model_name == 'SAS':
        from experiments.SAS_naive import SAS
        model_type = SAS
    elif model_name == 'TAS':
        from experiments.TAS_naive import TAS
        model_type = TAS
    elif model_name == 'OBA':
        from experiments.OBA_naive import OBA
        model_type = OBA
    elif model_name == 'FPN':
        from experiments.FPN_naive import FPN
        model_type = FPN
    elif model_name == 'FCN':
        from experiments.FCN_naive import FCN
        model_type = FCN     
    elif model_name == 'LSTM':
        from experiments.LSTM_naive import LSTM 
#         from experiments.Test_LSTM_naive import LSTM      
        
        model_type = LSTM
    elif model_name == 'MLP':
        from experiments.MLP_naive import MLP
        model_type = MLP         
    elif model_name == 'LR':
        from experiments.LR_naive import LR
        model_type = LR
    elif model_name == 'SVR':
        from experiments.SVR_naive import SVR
        model_type = SVR
    elif model_name == 'RF':
        from experiments.RF_naive import RF
        model_type = RF
    elif model_name == 'Bayes':
        from experiments.Bayesian_naive import Bayes
        model_type = Bayes            
    # Integrated Forecast System
    elif model_name == 'IFS':
        from experiments.IFS_naive import IFS
        model_type = IFS         
    else:
        raise NotImplementedError
    return model_type

parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', '-d', nargs='*', help='devices id to be used', type=int, default=list(range(torch.cuda.device_count())))
parser.add_argument('--model_idx', '-m', help='model index', type=int, required=True)
parser.add_argument('--config', '-c', help='config file', type=str, required=True)
parser.add_argument("--num_workers", help='num_workers', default=16, type=int)
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
#输出基目录
parser.add_argument("--basePath", default = "/mnt/pami14/yqliu/EC-BiasCorrecting")
args = parser.parse_args()
###########################

STAS_like_list = ['STAS', 'SAS', 'TAS', 'OBA', 'FPN', 'FCN', 'LSTM', 'MLP', 'LR', 'SVR', 'RF', 'Bayes', 'IFS']

if __name__ == '__main__':
    with open(args.config) as f:
        config = yaml.load(f)
    print ('local_rank:', args.local_rank)
    print ('device_ids:', args.device_ids)
    torch.cuda.set_device(args.device_ids[args.local_rank])
    args.model_name = STAS_like_list[args.model_idx]    
    model_type = get_model(args)
    
    from utils.util_main import get_file_name
    # 数据集名称基SHO/ERA
    args.dataset_name = get_file_name(args.config)
    print ('BaseDatasetName:', args.dataset_name)
    # 保存数据集下面某一个方法的输出目录
    args.save_root = osp.join(
        args.basePath,
        'outputs',
        args.dataset_name,
        model_type.__name__)                             
    params = config[args.model_name]
    if args.local_rank == 0:
        for k, v in config[args.model_name].items():
            print('{}: {}'.format(k, v))
        
    
    model = model_type(args, **params)
    model.init_module()
    model.fit()    