#!/bin/bash

dropbox=../../dropbox
inner_opt=RMSprop
mda_lr=0.0001
latent=64
unroll=3

save_dir=$HOME/scratch/results/cvb/celeb_parametric_opt/opt-${inner_opt}-l-${latent}-mda-${mda_lr}-unroll-${unroll}

export CUDA_VISIBLE_DEVICES=0

python celeb_parametric_opt.py \
    -save_dir $save_dir \
    -dropbox $dropbox \
    -init_model_dump $save_dir/epoch-74 \
    -unroll_test 1 \
    -vis_num 16 \
    -unroll_steps $unroll \
    -inner_opt $inner_opt \
    -img_size 64 \
    -mda_lr $mda_lr \
    -latent_dim $latent \
    -learning_rate 1e-3 \
    -ctx gpu
    $@
