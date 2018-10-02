#!/bin/bash

AVG_FILTER=$1
DENOISE=$2
DISCRIM=$3
PIPELINE_CONFIG_PATH=$4
TRAIN_PATH=$5
OUTPUT_PATH=$6

# From tensorflow/models/research/
python object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAIN_PATH} \
    --output_directory=${OUTPUT_PATH} \
    --average_filter=${AVG_FILTER} \
    --denoise=${DENOISE} \
    --discrim=${DISCRIM}
