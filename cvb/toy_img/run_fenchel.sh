#!/bin/bash

save_dir=$HOME/scratch/results/cvb/toy_img/nonparam_fenchel

if [ ! -e $save_dir ]; 
then
    mkdir -p $save_dir
fi

python -u nonparam_fenchel.py \
    -save_dir $save_dir \
    -learning_rate 2e-5 \
    -batch_size 500 \
    -mda_lr 0.001 \
    -unroll_steps 5 \
