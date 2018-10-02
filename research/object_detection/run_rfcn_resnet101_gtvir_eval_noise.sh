
# CUDA_VISIBLE_DEVICES=1 python eval.py --gaussian_noise=False --lowres=True --discrim=True --subsample_factor=8 --average_filter=True --denoise=True --stddev=0.0  --pipeline_config_path=samples/configs/rfcn_resnet101_gtvir.config --checkpoint_dir=checkpoints/rfcn_resnet101_gtvir_gaussian_hrlr8x_avgfilter_denoise_noddiscrim_discrim_entirefinetune --eval_dir=evals/rfcn_resnet101_gtvir_gaussian_hrlr8x_avgfilter_denoise_noddiscrim_discrim_entirefinetune_lr8x
# CUDA_VISIBLE_DEVICES=1 python eval.py --gaussian_noise=True --lowres=False --discrim=True --subsample_factor=8 --average_filter=True --denoise=True --stddev=0.15  --pipeline_config_path=samples/configs/rfcn_resnet101_merge_train.config --checkpoint_dir=checkpoints/GTVIR_Jun01 --eval_dir=evals/rfcn_resnet101_baseline_noise/eval/
#CUDA_VISIBLE_DEVICES=1 python eval.py --gaussian_noise=True --stddev=0.15  --pipeline_config_path=samples/configs/rfcn_resnet101_gtvir_merge_train.config --checkpoint_dir=checkpoints/GTVIR_Jun01 --eval_dir=evals/rfcn_resnet101_baseline_noise/eval/


CUDA_VISIBLE_DEVICES=1 python eval.py --gaussian_noise=True --stddev=0.15 --discrim=True --average_filter=True --denoise=True --pipeline_config_path=samples/configs/rfcn_resnet101_gtvir_merge_train.config --checkpoint_dir=checkpoints/rfcn_resnet101_gtvir_noise_merged_trained --eval_dir=evals/rfcn_resnet101_noiserobust_gtvir_noise/eval/
CUDA_VISIBLE_DEVICES=1 python eval.py --discrim=True --average_filter=True --denoise=True --pipeline_config_path=samples/configs/rfcn_resnet101_gtvir_merge_train.config --checkpoint_dir=checkpoints/rfcn_resnet101_gtvir_noise_merged_trained/ --eval_dir=evals/rfcn_resnet101_noiserobust_gtvir/eval/

