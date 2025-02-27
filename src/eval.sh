#!/bin/bash


PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python test.py \
--model_path /local2/amvepa91/RadFM/src/gli_run/checkpoint-62208 \
--test_data_path /local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_v3.json --output_dir ./gli_run

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python test.py \
--model_path /local2/amvepa91/RadFM/src/met_run/checkpoint-18180 \
--test_data_path /local2/amvepa91/MedTrinity-25M/brats_met_3d_vqa_subjTrue_test_v1.json --output_dir ./met_run

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python test.py \
--model_path /local2/amvepa91/RadFM/src/goat_run/checkpoint-38880 \
--test_data_path /local2/amvepa91/MedTrinity-25M/brats_goat_3d_vqa_subjTrue_test_v1.json --output_dir ./goat_run