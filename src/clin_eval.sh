#!/bin/bash


PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python test.py \
--model_path /local2/amvepa91/RadFM/src/gli_new_dataset_v10_run/checkpoint-93312/pytorch_model.bin \
--test_data_path /local2/amvepa91/clinical_validation_test_set.json --output_dir ./gli_new_dataset_v10_run