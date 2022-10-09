# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
python pretrain.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=8 batch_size=4\
    seed=11\
    epoch_num=50\
    cuda_device=$1\
    save_type=max_score\
    turn_level=True\
    only_target_loss=True\
    input_history=False\
    input_prev_resp=True\
    exp_no=MGA