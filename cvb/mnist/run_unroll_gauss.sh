#!/bin/bash

dropbox=../../dropbox
inner_opt=SGD
mda_lr=0.1
unroll=5
latent=32
arch=cnn

save_dir=$HOME/scratch/results/cvb/mnist_unroll_gauss/a-${arch}-opt-${inner_opt}-l-${latent}-mda-${mda_lr}-u-${unroll}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

log_file=$save_dir/log.txt

export CUDA_VISIBLE_DEVICES=0

python mnist_unroll_gauss.py \
    -save_dir $save_dir \
    -log_file $log_file \
    -arch $arch \
    -dropbox $dropbox \
    -unroll_test 1 \
    -binary 1 \
    -unroll_steps $unroll \
    -inner_opt $inner_opt \
    -img_size 32 \
    -mda_lr $mda_lr \
    -latent_dim $latent \
    -learning_rate 1e-4 \
    -num_epochs 1000 \
    -ctx gpu \
    $@
