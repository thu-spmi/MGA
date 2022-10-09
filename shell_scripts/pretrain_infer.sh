# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
python pretrain.py -mode pretrain\
    -cfg  lr=1e-4\
    seed=11\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=50\
    cuda_device=$1\
    spv_proportion=$2\
    posterior_train=True\
    save_type=min_loss\
    turn_level=True\
    input_history=False\
    input_prev_resp=True\
    exp_no=MGA-infer-${2}