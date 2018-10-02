# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Evaluation executable for detection models.

This executable is used to evaluate DetectionModels. There are two ways of
configuring the eval job.

1) A single pipeline_pb2.TrainEvalPipelineConfig file maybe specified instead.
In this mode, the --eval_training_data flag may be given to force the pipeline
to evaluate on training data instead.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files may be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being evaluated, an
input_reader_pb2.InputReader file to specify what data the model is evaluating
and an eval_pb2.EvalConfig file to configure evaluation parameters.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --eval_config_path=eval_config.pbtxt \
        --model_config_path=model_config.pbtxt \
        --input_config_path=eval_input_config.pbtxt
"""
import functools
import os
import tensorflow as tf

from object_detection import evaluator
from object_detection.builders import input_reader_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util


tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job.')
flags.DEFINE_string('checkpoint_dir', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.DEFINE_string('eval_dir', '/tmp/faster_rcnn_inception_v2_coco',
                    'Directory to write eval summaries to.')
flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_string('eval_config_path', '',
                    'Path to an eval_pb2.EvalConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')
flags.DEFINE_boolean('run_once', False, 'Option to only run a single pass of '
                     'evaluation. Overrides the `max_evals` parameter in the '
                     'provided config.')
flags.DEFINE_string('similarity_loss', 'bidirection',
                    'Choose between pivot and bidirection.')
flags.DEFINE_integer('res_depth', 0, 'Depth of the residual block for the generator')
flags.DEFINE_boolean('resize', False,
                    'True for resizing the image.')
flags.DEFINE_boolean('discrim', False,
                    'True for discrim.')
flags.DEFINE_boolean('denoise', False,
                    'True for denoise.')
flags.DEFINE_boolean('generator_separate_channel', False,
                    'True for channel wise learning.')
flags.DEFINE_boolean('denoise_discrim', False,
                     'True for enabling discriminator network.')
flags.DEFINE_integer('ks', 3,
                     'size of the convolutional filter')
flags.DEFINE_integer('ngf', 16,
                     '# of gen filters in first conv layer')
flags.DEFINE_integer('ndf', 16,
                     '# of discrim filters in first conv layer')
flags.DEFINE_float('stddev', 0.15,
                   'stddev for gaussian noise assuming pixels are'
                   'in range [0, 1)')
flags.DEFINE_float('ratio', 0.01,
                   'ratio for salt_pepper noise')
flags.DEFINE_integer('filter_size', 3,
                   'filter size for median filter or average filter')
flags.DEFINE_boolean('median_filter', False,
                     'True for median filter use.')
flags.DEFINE_boolean('average_filter', False,
                     'True for average filter use.')
flags.DEFINE_boolean('mixture_of_filters', False,
                     'True for mixture of filters.')
flags.DEFINE_boolean('salt_pepper_noise', False,
                     'True for original + salt_pepper noise training/evaluation.')
flags.DEFINE_boolean('gaussian_noise', False,
                     'True for original + gaussian noise training/evaluation.')
flags.DEFINE_boolean('snow', False,
                     'True for original + snow effect training.')
flags.DEFINE_boolean('lowres', False,
                     'True for low + high resolution training.')
flags.DEFINE_integer('subsample_factor', 4, 'Subsampling factor ')
flags.DEFINE_integer('resize_method', 0, 'Resize method '
                     '0: bilinear, 1: nearest_neighbor'
                     '2: bicubic, 3: area')
flags.DEFINE_boolean('upsample', True,
                     'True for low + high resolution training.')
FLAGS = flags.FLAGS


def main(unused_argv):
  assert FLAGS.checkpoint_dir, '`checkpoint_dir` is missing.'
  assert FLAGS.eval_dir, '`eval_dir` is missing.'
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  if FLAGS.pipeline_config_path:
    configs = config_util.get_configs_from_pipeline_file(
        FLAGS.pipeline_config_path)
    tf.gfile.Copy(FLAGS.pipeline_config_path,
                  os.path.join(FLAGS.eval_dir, 'pipeline.config'),
                  overwrite=True)
  else:
    configs = config_util.get_configs_from_multiple_files(
        model_config_path=FLAGS.model_config_path,
        eval_config_path=FLAGS.eval_config_path,
        eval_input_config_path=FLAGS.input_config_path)
    for name, config in [('model.config', FLAGS.model_config_path),
                         ('eval.config', FLAGS.eval_config_path),
                         ('input.config', FLAGS.input_config_path)]:
      tf.gfile.Copy(config,
                    os.path.join(FLAGS.eval_dir, name),
                    overwrite=True)

  model_config = configs['model']
  eval_config = configs['eval_config']
  input_config = configs['eval_input_config']

  model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=False)

  create_input_dict_fn = functools.partial(
      input_reader_builder.build,
      input_config)

  label_map = label_map_util.load_labelmap(input_config.label_map_path)
  max_num_classes = max([item.id for item in label_map.item])
  categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes)
  matching_iou_thresholds = [0.5]
#  matching_iou_thresholds = [i/100. for i in range(50, 100, 5)]

  if FLAGS.run_once:
    eval_config.max_evals = 1

  metrics = evaluator.evaluate(create_input_dict_fn, model_fn, eval_config, categories,
                     matching_iou_thresholds,
                     FLAGS.checkpoint_dir, FLAGS.eval_dir)

#  for key, value in sorted(metrics.iteritems()):
#    if 'PerformanceByCategory' in key:
#      print key, value
#  mAP_list = []
#  for key, value in sorted(metrics.iteritems()):
#    if 'mAP' in key:
#      print key, value
#      mAP_list.append(value)
#  print 'average mAP: ', sum(mAP_list)/len(mAP_list)

if __name__ == '__main__':
  tf.app.run()
