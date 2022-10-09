# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
path1=$3
path2=$4
python semi_train.py -mode semi_VL\
    -cfg PrioriModel_path=$path1 PosteriorModel_path=$path2\
    lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=40\
    cuda_device=$1\
    spv_proportion=$2\
    turn_level=True\
    exp_no=MGA-VL
