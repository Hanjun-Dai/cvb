#!/bin/bash

dropbox=../../dropbox
mda_lr=0.0001
latent=32
unroll=5
arch=cnn

save_dir=$HOME/scratch/results/cvb/mnist_gauss_fenchel/a-${arch}-l-${latent}-mda-${mda_lr}-unroll-${unroll}

## reload the model and finetune
#init_model_dump=$save_dir/epoch-700

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python mnist_gauss_fenchel.py \
    -save_dir $save_dir \
    -arch $arch \
    -dropbox $dropbox \
    -unroll_test 1 \
    -mda_decay_factor 1.0 \
    -unroll_steps $unroll \
    -num_epochs 700 \
    -img_size 32 \
    -mda_lr $mda_lr \
    -latent_dim $latent \
    -learning_rate 1e-4 \
    -binary 1 \
    -ctx gpu \
#    -init_model_dump $init_model_dump \
    $@
