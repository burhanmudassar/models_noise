
CUDA_VISIBLE_DEVICES=0 python train.py --pipeline_config_path=samples/configs/ssd_mobilenet_v1_coco.config --train_dir=checkpoints/ssd_mobilenet_v1_coco_train --lowres=True --subsample_factor=4 --resize=True
#CUDA_VISIBLE_DEVICES=1 python train.py --pipeline_config_path=samples/configs/ssd_mobilenet_v1_coco_200k.config --train_dir=checkpoints/ssd_mobilenet_v1_coco_train_10k --lowres=True --subsample_factor=4 --resize=True
#CUDA_VISIBLE_DEVICES=1 python train.py --pipeline_config_path=samples/configs/ssd_mobilenet_v1_coco_400k.config --train_dir=checkpoints/ssd_mobilenet_v1_coco_train --lowres=True --subsample_factor=4
#CUDA_VISIBLE_DEVICES=1 python train.py --pipeline_config_path=samples/configs/ssd_mobilenet_v1_coco_600k.config --train_dir=checkpoints/ssd_mobilenet_v1_coco_train --lowres=True --subsample_factor=4
