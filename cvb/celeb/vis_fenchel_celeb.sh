#!/bin/bash

dropbox=../../dropbox
mda_lr=0.0001
latent=64
unroll=3
lr=1e-3
save_dir=$HOME/scratch/results/cvb/celeb_fenchel/l-${latent}-mda-${mda_lr}-unroll-${unroll}-lr-${lr}

export CUDA_VISIBLE_DEVICES=0

python celeb_fenchel.py \
    -save_dir $save_dir \
    -dropbox $dropbox \
    -init_model_dump $save_dir/epoch-49 \
    -unroll_test 1 \
    -vis_num 16 \
    -unroll_steps $unroll \
    -img_size 64 \
    -mda_lr $mda_lr \
    -latent_dim $latent \
    -learning_rate $lr \
    -ctx gpu
    $@
