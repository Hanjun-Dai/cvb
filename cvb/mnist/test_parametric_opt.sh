#!/bin/bash

dropbox=../../dropbox
inner_opt=RMSprop
mda_lr=0.001
latent=32
unroll=3
arch=cnn

save_dir=$HOME/scratch/results/cvb/mnist_parametric_opt/a-${arch}-opt-${inner_opt}-l-${latent}-mda-${mda_lr}-unroll-${unroll}
init_model_dump=$save_dir/epoch-999

export CUDA_VISIBLE_DEVICES=0

python mnist_parametric_opt.py \
    -save_dir $save_dir \
    -arch $arch \
    -dropbox $dropbox \
    -test_is 1 \
    -vis_num 16 \
    -unroll_test 1 \
    -unroll_steps $unroll \
    -inner_opt $inner_opt \
    -img_size 32 \
    -mda_lr $mda_lr \
    -latent_dim $latent \
    -learning_rate 1e-4 \
    -binary 1 \
    -ctx gpu \
    -init_model_dump $init_model_dump \
    $@
