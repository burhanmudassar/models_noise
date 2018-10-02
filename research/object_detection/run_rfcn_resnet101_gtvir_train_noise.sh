#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python train.py --merge=True --gaussian_noise=True --discrim=True --denoise_discrim=False --average_filter=True --denoise=True --pipeline_config_path=samples/configs/rfcn_resnet101_gtvir_merge.config --preprocessing_checkpoint=preprocess_checkpoints/gaussian_avgfilter_denoise_discrim/model.ckpt-14518 --train_dir=checkpoints/rfcn_resnet101_gtvir_noise_merged/

mkdir checkpoints/rfcn_resnet101_gtvir_noise_merged_trained
cp checkpoints/rfcn_resnet101_gtvir_noise_merged/model.ckpt.* checkpoints/rfcn_resnet101_gtvir_noise_merged_trained/.
cp checkpoints/rfcn_resnet101_gtvir_noise_merged/checkpoint   checkpoints/rfcn_resnet101_gtvir_noise_merged_trained/.
CUDA_VISIBLE_DEVICES=1 python train.py --gaussian_noise=True --discrim=True --average_filter=True --denoise=True --pipeline_config_path=samples/configs/rfcn_resnet101_gtvir_merge_train.config --entire_finetune=True --train_dir=checkpoints/rfcn_resnet101_gtvir_noise_merged_trained
