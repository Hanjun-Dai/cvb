#!/bin/bash

dropbox=../../dropbox
mda_lr=0.0001
latent=64
unroll=3
lr=1e-4
save_dir=$HOME/scratch/results/cvb/celeb_fenchel/l-${latent}-mda-${mda_lr}-unroll-${unroll}-lr-${lr}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python celeb_fenchel.py \
    -save_dir $save_dir \
    -dropbox $dropbox \
    -unroll_test 1 \
    -unroll_steps $unroll \
    -num_epochs 500 \
    -epoch_save 10 \
    -img_size 64 \
    -mda_lr $mda_lr \
    -latent_dim $latent \
    -learning_rate $lr \
    -ctx gpu
    $@
