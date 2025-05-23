#!/bin/bash

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python test.py \
--model_path RadFM/src/gli_new_dataset_multimodal_run/checkpoint-93312/pytorch_model.bin \
--test_data_path brats_gli_3d_vqa_subjTrue_test_updated_v2_seed0_multitask_fixed.json \
--output_dir ./gli_new_dataset_multimodal_run

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python test.py \
--model_path RadFM/src/met_new_dataset_v3_run/checkpoint-27270/pytorch_model.bin \
--test_data_path brats_met_3d_vqa_subjTrue_test_updated_v3_seed0_multitask_fixed.json \
--output_dir ./met_new_dataset_v3_run

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 python test.py \
--model_path RadFM/src/goat_new_dataset_v3_run/checkpoint-38880/pytorch_model.bin \
--test_data_path brats_goat_3d_vqa_subjTrue_test_updated_v3_seed0_multitask_fixed.json \
--output_dir ./goat_new_dataset_v3_run