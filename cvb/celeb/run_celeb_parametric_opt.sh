#!/bin/bash

dropbox=../../dropbox
inner_opt=RMSprop
mda_lr=0.0001
latent=64
unroll=3

save_dir=$HOME/scratch/results/cvb/celeb_parametric_opt/opt-${inner_opt}-l-${latent}-mda-${mda_lr}-unroll-${unroll}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python celeb_parametric_opt.py \
    -save_dir $save_dir \
    -dropbox $dropbox \
    -unroll_test 1 \
    -unroll_steps $unroll \
    -inner_opt $inner_opt \
    -img_size 64 \
    -mda_lr $mda_lr \
    -latent_dim $latent \
    -learning_rate 1e-3 \
    -ctx gpu
    $@
