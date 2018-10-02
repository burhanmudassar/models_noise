# LR Evaluation
CUDA_VISIBLE_DEVICES=1 python eval.py --lowres=True --subsample_factor=4 --resize_method=0 --logtostderr --pipeline_config_path=samples/configs/ssd_mobilenet_v1_coco.config --checkpoint_dir=checkpoints/ssd_mobilenet_v1_coco_train --eval_dir=evals/ssd_mobilenet_v1_coco_lrx4 > log/ssd_mobilenet_v1_coco_lrx4.log
CUDA_VISIBLE_DEVICES=1 python eval.py --lowres=False --subsample_factor=4 --resize_method=0 --logtostderr --pipeline_config_path=samples/configs/ssd_mobilenet_v1_coco.config --checkpoint_dir=checkpoints/ssd_mobilenet_v1_coco_train --eval_dir=evals/ssd_mobilenet_v1_coco > log/ssd_mobilenet_v1_coco.log
# Clean Evaluation
#CUDA_VISIBLE_DEVICES=1 python eval.py --logtostderr --pipeline_config_path=samples/configs/ssd_mobilenet_v1_coco_600k.config --checkpoint_dir=checkpoints/ssd_mobilenet_v1_coco_train --eval_dir=evals/ssd_mobilenet_v1_coco_lrx4_highres > log/ssd_mobilenet_v1_coco_lrx4.log

# Original Model Evaluation
#CUDA_VISIBLE_DEVICES=1 python eval.py --logtostderr --pipeline_config_path=samples/configs/ssd_mobilenet_v1_coco_600k.config --checkpoint_dir=ssd_mobilenet_v1_coco_2018_01_28 --eval_dir=evals/ssd_mobilenet_v1_coco_orig_highres > log/ssd_mobilenet_v1_coco_orig_highres.log
#CUDA_VISIBLE_DEVICES=1 python eval.py --lowres=True --subsample_factor=4 --resize_method=0 --logtostderr --pipeline_config_path=samples/configs/ssd_mobilenet_v1_coco_600k.config --checkpoint_dir=ssd_mobilenet_v1_coco_2018_01_28 --eval_dir=evals/ssd_mobilenet_v1_coco_orig_lowres > log/ssd_mobilenet_v1_coco_orig_lowres.log