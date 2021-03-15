export OMP_NUM_THREADS=8
#> apex_oba.out 2>&1

# SSAS
python -m torch.distributed.launch --nproc_per_node=2 --master_port=88889 main.py -d 7 8 -m 0 -c ./config/SHO.yaml

# SAS
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=68001 main.py -d 1 2 -m 1 -c ./config/SHO.yaml

# STS
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=68003 main.py -d 1 2 -m 2 -c ./config/SHO.yaml

# OBA
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=48000 main.py -d 7 8 -m 3 -c ./config/SHO.yaml

# FPN
# python main.py -d 1 2 -m 4 -c ./config/SHO.yaml

# FCN
# python main.py -d 7 8 -m 5 -c ./config/SHO.yaml

# LSTM
# python main.py -d 1 2 -m 6 -c ./config/SHO.yaml 

# MLP
# python main.py -d 7 8 -m 7 -c ./config/SHO.yaml

# LR
# python main.py -d 7 8 -m 8 -c ./config/SHO.yaml

# SVR
# python main.py -d 7 8 -m 9 -c ./config/SHO.yaml

# RF
# python main.py -d 7 8 -m 10 -c ./config/SHO.yaml

# Bayesian
# python main.py -d 7 8 -m 11 -c ./config/SHO.yaml

# IFS
# python main.py -d 7 8 -m 12 -c ./config/SHO.yaml

# TPN
# python main.py -d 7 8 -m 13 -c ./config/SHO.yaml

# T-GCN
# python main.py -d 7 8 -m 14 -c ./config/SHO.yaml

# DA-RNN
# python main.py -d 7 8 -m 15 -c ./config/SHO.yaml



