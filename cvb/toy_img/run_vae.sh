#!/bin/bash

save_dir=$HOME/scratch/results/neural_opt/toy_img/vae

if [ ! -e $save_dir ]; 
then
    mkdir -p $save_dir
fi

python -u vae.py \
    -save_dir $save_dir \
    -log_file $save_dir/log.txt \
    -learning_rate 2e-5 \
    -batch_size 500 \
