#!/bin/bash


PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python train.py \
--train_data_path /local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_train_updated_v11_seed0_multitask_fixed.json \
--val_data_path /local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_val_updated_v11_seed0_multitask_fixed.json \
--output_dir ./gli_new_dataset_v11_run