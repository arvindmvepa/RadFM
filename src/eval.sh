#!/bin/bash


PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python test.py \
--test_data_path /local2/amvepa91/MedTrinity-25M/brats_gli_3d_vqa_subjTrue_test_v3.json --output_dir ./gli_run

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python test.py \
--test_data_path /local2/amvepa91/MedTrinity-25M/brats_met_3d_vqa_subjTrue_test_v1.json --output_dir ./met_run

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python test.py \
--test_data_path /local2/amvepa91/MedTrinity-25M/brats_goat_3d_vqa_subjTrue_test_v1.json --output_dir ./goat_run