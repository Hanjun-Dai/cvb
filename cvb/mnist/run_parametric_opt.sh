#!/bin/bash

dropbox=../../dropbox
inner_opt=RMSprop
mda_lr=0.001
latent=32
unroll=3
arch=cnn

save_dir=$HOME/scratch/results/cvb/mnist_parametric_opt/a-${arch}-opt-${inner_opt}-l-${latent}-mda-${mda_lr}-unroll-${unroll}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

log_file=$save_dir/log.txt

export CUDA_VISIBLE_DEVICES=0

python mnist_parametric_opt.py \
    -save_dir $save_dir \
    -arch $arch \
    -log_file $log_file \
    -dropbox $dropbox \
    -unroll_test 1 \
    -unroll_steps $unroll \
    -inner_opt $inner_opt \
    -img_size 32 \
    -mda_lr $mda_lr \
    -latent_dim $latent \
    -learning_rate 1e-4 \
    -binary 1 \
    -num_epochs 1000 \
    -ctx gpu \
    $@
