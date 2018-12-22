#!/bin/bash

dropbox=../../dropbox
mda_lr=0.0001
latent=32
unroll=5
arch=cnn

save_dir=$HOME/scratch/results/cvb/mnist_gauss_fenchel/a-${arch}-l-${latent}-mda-${mda_lr}-unroll-${unroll}
init_model_dump=$save_dir/epoch-290

export CUDA_VISIBLE_DEVICES=0

python mnist_gauss_fenchel.py \
    -save_dir $save_dir \
    -arch $arch \
    -dropbox $dropbox \
    -test_is 1 \
    -vis_num 0 \
    -unroll_test 1 \
    -unroll_steps $unroll \
    -binary 1 \
    -img_size 32 \
    -mda_lr $mda_lr \
    -latent_dim $latent \
    -ctx cpu \
    -init_model_dump $init_model_dump \
    $@
