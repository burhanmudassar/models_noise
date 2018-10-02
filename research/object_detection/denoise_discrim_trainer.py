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

"""Detection model trainer.

This file provides a generic training method that can be used to train a
DetectionModel.
"""

import functools

import tensorflow as tf

from object_detection.builders import optimizer_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import batcher
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.utils import variables_helper
from deployment import model_deploy
#from denoise import denoise, ae, discrim, discrim_loss
from collections import namedtuple
from module import *
import re
import time
TOWER_NAME = 'tower'

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

def activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def optimistic_restore(session, save_file):
  reader = tf.train.NewCheckpointReader(save_file)
  saved_shapes = reader.get_variable_to_shape_map()
  var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                  if var.name.split(':')[0] in saved_shapes])
  restore_vars = []
  name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
  with tf.variable_scope('', reuse=True):
    for var_name, saved_var_name in var_names:
      curr_var = name2var[saved_var_name]
      var_shape = curr_var.get_shape().as_list()
      if var_shape == saved_shapes[saved_var_name]:
        restore_vars.append(curr_var)
  saver = tf.train.Saver(restore_vars)
  saver.restore(session, save_file)


def create_input_queue(batch_size_per_clone, create_tensor_dict_fn,
                       batch_queue_capacity, num_batch_queue_threads,
                       prefetch_queue_capacity, data_augmentation_options):
  """Sets up reader, prefetcher and returns input queue.

  Args:
    batch_size_per_clone: batch size to use per clone.
    create_tensor_dict_fn: function to create tensor dictionary.
    batch_queue_capacity: maximum number of elements to store within a queue.
    num_batch_queue_threads: number of threads to use for batching.
    prefetch_queue_capacity: maximum capacity of the queue used to prefetch
                             assembled batches.
    data_augmentation_options: a list of tuples, where each tuple contains a
      data augmentation function and a dictionary containing arguments and their
      values (see preprocessor.py).

  Returns:
    input queue: a batcher.BatchQueue object holding enqueued tensor_dicts
      (which hold images, boxes and targets).  To get a batch of tensor_dicts,
      call input_queue.Dequeue().
  """
  tensor_dict = create_tensor_dict_fn()

  tensor_dict[fields.InputDataFields.image] = tf.expand_dims(
      tensor_dict[fields.InputDataFields.image], 0)

  images = tensor_dict[fields.InputDataFields.image]
  float_images = tf.to_float(images)
  tensor_dict[fields.InputDataFields.image] = float_images

  include_instance_masks = (fields.InputDataFields.groundtruth_instance_masks
                            in tensor_dict)
  include_keypoints = (fields.InputDataFields.groundtruth_keypoints
                       in tensor_dict)
  if data_augmentation_options:
    tensor_dict = preprocessor.preprocess(
        tensor_dict, data_augmentation_options,
        func_arg_map=preprocessor.get_default_func_arg_map(
            include_instance_masks=include_instance_masks,
            include_keypoints=include_keypoints))

  input_queue = batcher.BatchQueue(
      tensor_dict,
      batch_size=batch_size_per_clone,
      batch_queue_capacity=batch_queue_capacity,
      num_batch_queue_threads=num_batch_queue_threads,
      prefetch_queue_capacity=prefetch_queue_capacity)
  return input_queue


def get_inputs(input_queue, num_classes, merge_multiple_label_boxes=False):
  """Dequeues batch and constructs inputs to object detection model.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    num_classes: Number of classes.
    merge_multiple_label_boxes: Whether to merge boxes with multiple labels
      or not. Defaults to false. Merged boxes are represented with a single
      box and a k-hot encoding of the multiple labels associated with the
      boxes.

  Returns:
    images: a list of 3-D float tensor of images.
    image_keys: a list of string keys for the images.
    locations_list: a list of tensors of shape [num_boxes, 4]
      containing the corners of the groundtruth boxes.
    classes_list: a list of padded one-hot tensors containing target classes.
    masks_list: a list of 3-D float tensors of shape [num_boxes, image_height,
      image_width] containing instance masks for objects if present in the
      input_queue. Else returns None.
    keypoints_list: a list of 3-D float tensors of shape [num_boxes,
      num_keypoints, 2] containing keypoints for objects if present in the
      input queue. Else returns None.
  """
  read_data_list = input_queue.dequeue()
  label_id_offset = 1
  def extract_images_and_targets(read_data):
    """Extract images and targets from the input dict."""
    image = read_data[fields.InputDataFields.image]
    key = ''
    if fields.InputDataFields.source_id in read_data:
      key = read_data[fields.InputDataFields.source_id]
    location_gt = read_data[fields.InputDataFields.groundtruth_boxes]
    classes_gt = tf.cast(read_data[fields.InputDataFields.groundtruth_classes],
                         tf.int32)
    classes_gt -= label_id_offset
    if merge_multiple_label_boxes:
      location_gt, classes_gt, _ = util_ops.merge_boxes_with_multiple_labels(
          location_gt, classes_gt, num_classes)
    else:
      classes_gt = util_ops.padded_one_hot_encoding(
          indices=classes_gt, depth=num_classes, left_pad=0)
    masks_gt = read_data.get(fields.InputDataFields.groundtruth_instance_masks)
    keypoints_gt = read_data.get(fields.InputDataFields.groundtruth_keypoints)
    if (merge_multiple_label_boxes and (
        masks_gt is not None or keypoints_gt is not None)):
      raise NotImplementedError('Multi-label support is only for boxes.')
    return image, key, location_gt, classes_gt, masks_gt, keypoints_gt

  return zip(*map(extract_images_and_targets, read_data_list))

def get_salt_pepper_noise_image(image, ratio=0.01, rand_ratio=False):

  image_shape = tf.shape(image)
  num_batch = image_shape[0]
  height = image_shape[1]
  width = image_shape[2]

  # uniform [1 - ratio, 2 - ratio)
  ratio = tf.random_uniform([], maxval=ratio) if rand_ratio else ratio
  random_tensor = 1. - ratio
  random_tensor += tf.random_uniform(image_shape)
  # 0: ratio, 1: 1 - ratio
  binary_tensor = tf.floor(random_tensor)

  # scale image from [-1, 1] to [0, 1]
  random_tensor2 = tf.random_uniform(image_shape)
  image = (image + 1.)/2.
  noise_image = binary_tensor*image + (1.-binary_tensor)*random_tensor2

  # scaling pixel values to be in [-1, 1]).
  noise_image = tf.clip_by_value(noise_image*2.-1, -1, 1)
  return noise_image

def get_gaussian_noise_image(image, stddev=0.15, rand_stddev=False):

  image_shape = tf.shape(image)
  num_batch = image_shape[0]
  height = image_shape[1]
  width = image_shape[2]
  
  if rand_stddev:
    sigma = tf.abs(tf.truncated_normal([num_batch], mean=0., stddev=stddev/2))
  else:
    sigma = stddev
  # since preprocessed input values are in [-1, 1], we use sigma*2
  noise = tf.random_normal(image_shape, mean=0., stddev=sigma*2)
  # scaling pixel values to be in [-1, 1]).
  noise_image = tf.clip_by_value(image + noise, -1, 1)
  return noise_image

def get_snow_image(image, sparsity=0.05):
  # image shape: [1, None, None, 3]

  image_shape = image.get_shape().as_list()
  height = image_shape[1]
  width = image_shape[2]
  
  # assume the size of the snow ball is 24.
  #    **
  #   ****
  #  ******
  #  ******
  #   ****
  #    **
  # based on sparsity we calculate the number of snow balls in a image
  num_snows = int(height*width*sparsity/24)
  
  snow_rows = tf.random_uniform([num_snows], 0, height, dtype=tf.int32)
  snow_cols = tf.random_uniform([num_snows], 0, width, dtype=tf.int32)

def resize_dividable_image(image, number):
  # image shape: [1, None, None, 3]

  image_shape = tf.shape(image)
  height = image_shape[1]/number*number
  width = image_shape[2]/number*number
  return tf.image.resize_images(image, [height, width])
  

def get_lowres_image(image):
  # image shape: [1, None, None, 3]

  image_shape = tf.shape(image)
  height = image_shape[1]
  width = image_shape[2]

  # randomly select subsample factor to be 2 or 4
  subsample = tf.floor(tf.random_uniform([], 0, 2))
  down_height, down_width = tf.cond(tf.equal(subsample, 1),
                              lambda: (tf.to_int32(height/2), tf.to_int32(width/2)),
                              lambda: (tf.to_int32(height/4), tf.to_int32(width/4)))
  downsampled_image = tf.image.resize_images(image,
                                        [down_height, down_width])

  # randomly select resize methods
  resize_method = tf.to_int32(tf.floor(tf.random_uniform([], 0, 4)))

  upsampled_image0 = tf.image.resize_images(downsampled_image,
                                           [height, width], method=0)
  upsampled_image1 = tf.image.resize_images(downsampled_image,
                                           [height, width], method=1)
  upsampled_image2 = tf.image.resize_images(downsampled_image,
                                           [height, width], method=2)
  upsampled_image3 = tf.image.resize_images(downsampled_image,
                                           [height, width], method=3)
  upsampled_image = tf.case({tf.equal(resize_method, 0): lambda: upsampled_image0,
                             tf.equal(resize_method, 1): lambda: upsampled_image1,
                             tf.equal(resize_method, 2): lambda: upsampled_image2,
                            }, default=lambda: upsampled_image3)
  return upsampled_image

def add_noise(images,
              salt_pepper_noise=False,
              gaussian_noise=False,
              lowres=False,
              snow=False):
  """Creates loss function for a DetectionModel.

  Args:
    images: list of image tensors.
    FLAGS: tf.app.flags.FLAGS
  """
  # After model's preprocess is done,
  # we double the size of the input mini batch for similarity learning
  noisy_images = images
  if salt_pepper_noise:
    noisy_images = [get_salt_pepper_noise_image(image, ratio=FLAGS.ratio, rand_ratio=True)
                    for image in noisy_images]
    tf.summary.image('salt_pepper_noise_images', noisy_images[0])
  if gaussian_noise:
    noisy_images = [get_gaussian_noise_image(image, FLAGS.stddev, rand_stddev=FLAGS.rand_stddev)
                    for image in noisy_images]
    tf.summary.image('gaussian_noise_images', noisy_images[0])
  if lowres:
    noisy_images = [get_lowres_image(image) for image in noisy_images]
    tf.summary.image('mixed_resolution_images', noisy_images[0])
  if snow:
    noisy_images = [get_snow_image(image) for image
                   in noisy_images]
    tf.summary.image('snow_images', noisy_images[0])

  return noisy_images

def gated_denoise(images, is_training, train_config):
  # Discriminator Network
  # We are just mixing the two images
  images_shape = images.get_shape().as_list()
  noisy_batch_size = train_config.batch_size if (FLAGS.lowres or FLAGS.gaussian_noise) else 0
  losses_dict = {}
  summaries_dict = {}

  OPTIONS = namedtuple('OPTIONS', 'gf_dim df_dim output_c_dim is_training')
  options = OPTIONS._make((FLAGS.ngf, FLAGS.ndf, 3, is_training))
  criterionGAN = mae_criterion
  criterionGAN2 = sce_criterion
  
  noisy_images = images[:noisy_batch_size]
  original_images = images[noisy_batch_size:]

  # Filter
  summaries_dict['prefiltered_noisy_images'] = tf.summary.image('prefiltered_noisy_images', images, max_outputs=1)
  summaries_dict['prefiltered_clean_images'] = tf.summary.image('prefiltered_clean_images', images[noisy_batch_size:], max_outputs=1)
  filtered_images = images
  if FLAGS.average_filter:
    if FLAGS.mixture_of_filters:
      with tf.variable_scope('filter_gate') as scope:
        print('Building mixture of filters model')
        filtered_images0 = average_filter(images, 2)
        filtered_images1 = average_filter(images, 3)
        filtered_images2 = average_filter(images, 4)
        probs = gate(images, FLAGS.ndf, num_classes=4, reuse=False, name='gate')
        activation_summary(probs)
        filtered_images = tf.add_n([probs[:,0,None,None,None]*images,
                                    probs[:,1,None,None,None]*filtered_images0,
                                    probs[:,2,None,None,None]*filtered_images1,
                                    probs[:,3,None,None,None]*filtered_images2])
    else:
      filtered_images = average_filter(images, FLAGS.filter_size)
    summaries_dict['filtered_noisy_images'] = tf.summary.image('filtered_noisy_images', filtered_images, max_outputs=1)
    summaries_dict['filtered_clean_images'] = tf.summary.image('filtered_clean_images', filtered_images[noisy_batch_size:], max_outputs=1)

  if FLAGS.srcnn:
    with tf.variable_scope('srcnn') as scope:
      if FLAGS.generator_separate_channel:
        denoised_images = generator_separate_resnet(images, options, res_depth=FLAGS.res_depth, reuse=False, name='generator') 
      else:
        denoised_images = generator_resnet(images, options, res_depth=FLAGS.res_depth, output_c_dim=3, reuse=False, name='generator') 
      denoise_sim_loss = abs_criterion(denoised_images[:noisy_batch_size], original_images, name='g_loss/sim_loss')
      g_loss = FLAGS.denoise_loss_factor * denoise_sim_loss
      losses_dict['g_loss'] = g_loss
      summaries_dict['g_loss/sim_loss'] = tf.summary.scalar('g_loss/sim_loss', denoise_sim_loss)
      if FLAGS.denoise_discrim:
        noisy_and_denoised = tf.concat([noisy_images, denoised_images[:noisy_batch_size]], 3)
        noisy_and_original = tf.concat([noisy_images, original_images], 3)
#        d_denoised = discriminator(noisy_and_denoised, FLAGS.ndf, reuse=False, name='discriminator')
#        d_original = discriminator(noisy_and_original, FLAGS.ndf, reuse=True,  name='discriminator')
        d_denoised = discriminator(denoised_images[:noisy_batch_size], FLAGS.ndf, FLAGS.ks, reuse=False, name='discriminator')
        d_original = discriminator(original_images, FLAGS.ndf, FLAGS.ks, reuse=True,  name='discriminator')

        # generator loss
        g_gan_loss = criterionGAN(d_denoised, tf.ones_like(d_denoised))
        summaries_dict['g_gan_loss'] = tf.summary.scalar('g_loss/g_gan_loss', g_gan_loss)

        g_loss += FLAGS.denoise_gan_loss_factor * g_gan_loss
        losses_dict['g_loss'] = g_loss

        # discriminator loss
        d_loss_real = criterionGAN(d_original, tf.ones_like(d_original))
        d_loss_fake = criterionGAN(d_denoised, tf.zeros_like(d_denoised))

        d_loss = FLAGS.denoise_gan_loss_factor * (d_loss_real + d_loss_fake) / 2 
        losses_dict['d_loss'] = d_loss

        summaries_dict['d_loss'] = tf.summary.scalar('d_loss', d_loss)

      summaries_dict['g_loss'] = tf.summary.scalar('g_loss', g_loss)
  
    filtered_images = denoised_images
    summaries_dict['denoised_noisy_images'] = tf.summary.image('denoised_noisy_images', filtered_images, max_outputs=1)
    summaries_dict['denoised_clean_images'] = tf.summary.image('denoised_clean_images', filtered_images[noisy_batch_size:], max_outputs=1)

  if FLAGS.denoise:
    with tf.variable_scope('denoise') as scope:
      if FLAGS.generator_separate_channel:
        denoised_images = generator_separate_resnet(filtered_images, options, res_depth=FLAGS.res_depth, reuse=False, name='generator') 
      else:
        denoised_images = generator_resnet(filtered_images, options, res_depth=FLAGS.res_depth, output_c_dim=3, reuse=False, name='generator') 
      denoise_sim_loss = abs_criterion(denoised_images[:noisy_batch_size], original_images, name='g_loss/sim_loss')
      g_loss = FLAGS.denoise_loss_factor * denoise_sim_loss
      losses_dict['g_loss'] = g_loss
      summaries_dict['g_loss/sim_loss'] = tf.summary.scalar('g_loss/sim_loss', denoise_sim_loss)
      if FLAGS.denoise_discrim:
        noisy_and_denoised = tf.concat([noisy_images, denoised_images[:noisy_batch_size]], 3)
        noisy_and_original = tf.concat([noisy_images, original_images], 3)
#        d_denoised = discriminator(noisy_and_denoised, FLAGS.ndf, reuse=False, name='discriminator')
#        d_original = discriminator(noisy_and_original, FLAGS.ndf, reuse=True,  name='discriminator')
        d_denoised = discriminator(denoised_images[:noisy_batch_size], FLAGS.ndf, FLAGS.ks, reuse=False, name='discriminator')
        d_original = discriminator(original_images, FLAGS.ndf, FLAGS.ks, reuse=True,  name='discriminator')

        # generator loss
        g_gan_loss = criterionGAN(d_denoised, tf.ones_like(d_denoised))
        summaries_dict['g_gan_loss'] = tf.summary.scalar('g_loss/g_gan_loss', g_gan_loss)

        g_loss += FLAGS.denoise_gan_loss_factor * g_gan_loss
        losses_dict['g_loss'] = g_loss

        # discriminator loss
        d_loss_real = criterionGAN(d_original, tf.ones_like(d_original))
        d_loss_fake = criterionGAN(d_denoised, tf.zeros_like(d_denoised))

        d_loss = FLAGS.denoise_gan_loss_factor * (d_loss_real + d_loss_fake) / 2 
        losses_dict['d_loss'] = d_loss

        summaries_dict['d_loss'] = tf.summary.scalar('d_loss', d_loss)

      summaries_dict['g_loss'] = tf.summary.scalar('g_loss', g_loss)
  
    filtered_images = denoised_images
    summaries_dict['denoised_noisy_images'] = tf.summary.image('denoised_noisy_images', filtered_images, max_outputs=1)
    summaries_dict['denoised_clean_images'] = tf.summary.image('denoised_clean_images', filtered_images[noisy_batch_size:], max_outputs=1)

  # Gate Operation
  if FLAGS.discrim:
    with tf.variable_scope('input_discrim') as scope:
      d_in_logits = discriminator(images, FLAGS.ndf, FLAGS.ks, reuse=False, name='discriminator')

      d_in_loss_real = criterionGAN2(d_in_logits[noisy_batch_size:],  tf.ones_like(d_in_logits[noisy_batch_size:]))
      d_in_loss_fake = criterionGAN2(d_in_logits[:noisy_batch_size], tf.zeros_like(d_in_logits[:noisy_batch_size]))

      d_in_loss = FLAGS.discrim_loss_factor * (d_in_loss_real + d_in_loss_fake) / 2 
      losses_dict['d_in_loss'] = d_in_loss
      summaries_dict['d_in_loss'] = tf.summary.scalar('d_in_loss', d_in_loss)
      #tf.add_to_collection('losses', losses_dict['d_in_loss'])
  
      d_in_logits = tf.reduce_mean(d_in_logits, [1, 2, 3])
      d_in_sigmoid = tf.nn.sigmoid(d_in_logits, name='is_clean')
      activation_summary(d_in_sigmoid)
      images =  tf.add(d_in_sigmoid[:,None,None,None] * images,
                       (1 - d_in_sigmoid[:,None,None,None]) * filtered_images)
  else:
    images = filtered_images

  summaries_dict['preprocessed_noisy_images'] = tf.summary.image('preprocessed_noisy_images', images, max_outputs=1)
  summaries_dict['preprocessed_clean_images'] = tf.summary.image('preprocessed_clean_images', images[noisy_batch_size:], max_outputs=1)

  return images, losses_dict, summaries_dict


def _create_losses(input_queue, create_model_fn, train_config):
  """Creates loss function for a DetectionModel.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    create_model_fn: A function to create the DetectionModel.
    train_config: a train_pb2.TrainConfig protobuf.
  """
  detection_model = create_model_fn()
  (images, _, groundtruth_boxes_list, groundtruth_classes_list,
   groundtruth_masks_list, groundtruth_keypoints_list) = get_inputs(
       input_queue,
       detection_model.num_classes,
       train_config.merge_multiple_label_boxes)
  images = [detection_model.preprocess(image) for image in images]

  # lowresolution injection
  tf.summary.image('model_preprocessed_images', images[0])

  # images[0] shape: (1, ?, ?, 3)
  # groundtruth_boxes_list shape: (?, 4)
  image_with_box = tf.image.draw_bounding_boxes(images[0],
                        tf.expand_dims(groundtruth_boxes_list[0], 0))
  tf.summary.image('image_with_bounding_boxes', image_with_box)

  noise = FLAGS.gaussian_noise or FLAGS.salt_pepper_noise
  distort = FLAGS.lowres or FLAGS.snow
  if noise or distort:
    # add noise
    noisy_images = add_noise(images,
                             gaussian_noise=FLAGS.gaussian_noise,
                             salt_pepper_noise=FLAGS.salt_pepper_noise)
    if FLAGS.additive_noise:
      noisy_images = add_noise(noisy_images,
                               lowres=FLAGS.lowres,
                               snow=FLAGS.snow)
    else:
      lowres_images = add_noise(images,
                                lowres=FLAGS.lowres,
                                snow=FLAGS.snow)
      inject_clean_or_noise = tf.floor(tf.random_uniform([], 0, 2))
      images = [tf.cond(tf.equal(inject_clean_or_noise, 0),
                       lambda: image,
                       lambda: lowres_image) for image, lowres_image in zip(images, lowres_images)]
    print('doubling input mini batch')
    print(images)
    print(noisy_images)
    images = noisy_images + images
    groundtruth_boxes_list += groundtruth_boxes_list
    groundtruth_classes_list += groundtruth_classes_list
    groundtruth_masks_list += groundtruth_masks_list

    # make sure to match the size of the images for denoise filter training
    # resize the images only when training denoise network.
    if FLAGS.resize:
      down_height = 600
      down_width = 600
      images = [tf.image.resize_images(image,
                [down_height, down_width])
                for image in images]
    else:
      images = [resize_dividable_image(image, 4) for image in images]

  # list of tensors --> tensors
  images = tf.concat(images, 0)

  # Gated denoise
  images, losses_dict, summaries_dict = gated_denoise(images, detection_model._is_training, train_config)


  # Denoise network
#  prediction_dict = detection_model.denoise(images)
#  denoised_images = prediction_dict['denoised_images']
#  losses_dict = detection_model.denoise_loss(prediction_dict)
#
#  tf.summary.image('denoised_images', denoised_images)
#  images = denoised_images


  if any(mask is None for mask in groundtruth_masks_list):
    groundtruth_masks_list = None
  if any(keypoints is None for keypoints in groundtruth_keypoints_list):
    groundtruth_keypoints_list = None

#  detection_model.provide_groundtruth(groundtruth_boxes_list,
#                                      groundtruth_classes_list,
#                                      groundtruth_masks_list,
#                                      groundtruth_keypoints_list)
#  prediction_dict.update(detection_model.predict(images))
#  losses_dict = detection_model.loss(prediction_dict)

  # added for debugging
#  for loss_tensor, loss_key in zip(losses_dict.values(), losses_dict.keys()):
  for loss_key, loss_tensor in losses_dict.items():
    print(loss_key)
    print(loss_tensor)
    tf.losses.add_loss(loss_tensor)
  return losses_dict, summaries_dict

def train(create_tensor_dict_fn, create_model_fn, train_config, master, task,
          num_clones, worker_replicas, clone_on_cpu, ps_tasks, worker_job_name,
          is_chief, train_dir):
  """Training function for detection models.

  Args:
    create_tensor_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel and generates
                     losses.
    train_config: a train_pb2.TrainConfig protobuf.
    master: BNS name of the TensorFlow master to use.
    task: The task id of this training instance.
    num_clones: The number of clones to run per machine.
    worker_replicas: The number of work replicas to train with.
    clone_on_cpu: True if clones should be forced to run on CPU.
    ps_tasks: Number of parameter server tasks.
    worker_job_name: Name of the worker job.
    is_chief: Whether this replica is the chief replica.
    train_dir: Directory to write checkpoints and training summaries to.
  """

  detection_model = create_model_fn()
  data_augmentation_options = [
      preprocessor_builder.build(step)
      for step in train_config.data_augmentation_options]

  with tf.Graph().as_default():
    # Build a configuration specifying multi-GPU and multi-replicas.
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=worker_replicas,
        num_ps_tasks=ps_tasks,
        worker_job_name=worker_job_name)

    # Place the global step on the device storing the variables.
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    with tf.device(deploy_config.inputs_device()):
      input_queue = create_input_queue(
          train_config.batch_size // num_clones, create_tensor_dict_fn,
          train_config.batch_queue_capacity,
          train_config.num_batch_queue_threads,
          train_config.prefetch_queue_capacity, data_augmentation_options)

    # Gather initial summaries.
    # TODO(rathodv): See if summaries can be added/extracted from global tf
    # collections so that they don't have to be passed around.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    global_summaries = set([])

    model_fn = functools.partial(_create_losses,
                                 create_model_fn=create_model_fn,
                                 train_config=train_config)
    clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
    first_clone_scope = clones[0].scope
    print clones

    losses_dict, summaries_dict = clones[0].outputs

#    # Soft placement allows placing on CPU ops without GPU implementation.
#    session_config = tf.ConfigProto(allow_soft_placement=True,
#                                    log_device_placement=False)
#    session_config.gpu_options.allow_growth = True

    print('global_variables')
    for var in tf.global_variables():
      print(var)

    d_opt = tf.train.AdamOptimizer()
    g_opt = tf.train.AdamOptimizer()
    d_in_opt = tf.train.AdamOptimizer()

    # Add histograms for trainable variables.
    t_vars = tf.trainable_variables()
  
    summary_histogram_denoise_discrim = []
    summary_histogram_denoise_generator = []
    summary_histogram_input_discrim = []
    for var in t_vars:
      if 'denoise/discriminator' in var.name:
        summary_histogram_denoise_discrim.append(tf.summary.histogram(var.op.name, var))
      elif 'denoise/generator' in var.name:
        summary_histogram_denoise_generator.append(tf.summary.histogram(var.op.name, var))
      elif 'input_discrim' in var.name:
        summary_histogram_input_discrim.append(tf.summary.histogram(var.op.name, var))
    if summary_histogram_denoise_discrim != []:
      summaries_dict['denoise_discrim_histogram'] = tf.summary.merge(summary_histogram_denoise_discrim)
    if summary_histogram_denoise_generator != []:
      summaries_dict['denoise_generator_histogram'] = tf.summary.merge(summary_histogram_denoise_generator)
    if summary_histogram_input_discrim != []:
      summaries_dict['input_discrim_histogram'] = tf.summary.merge(summary_histogram_input_discrim)

    # Make d_train_op
    d_vars  = [var for var in t_vars if 'denoise/discriminator' in var.name]
    g_vars  = [var for var in t_vars if 'denoise/generator' in var.name]
    d_in_vars  = [var for var in t_vars if 'input_discrim' in var.name]
  
    d_train_op = d_opt.minimize(losses_dict['d_loss'], var_list=d_vars) if d_vars !=[] else None
    g_train_op = g_opt.minimize(losses_dict['g_loss'], var_list=g_vars) if g_vars !=[] else None
    d_in_train_op = d_in_opt.minimize(losses_dict['d_in_loss'], global_step=global_step, var_list=d_in_vars) if d_in_vars !=[] else None

    # Track the moving averages of all trainable variables.
    MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Collect update ops
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_op:
      updates = tf.tuple(update_op)
      print ("Update ops are:")
      for update in updates:
        print (update)
      with tf.control_dependencies([variables_averages_op] + update_op):
        train_op = tf.no_op(name='train')
    else:
      print ("There is no update ops")
      with tf.control_dependencies([variables_averages_op]):
        train_op = tf.no_op(name='train')

    #train_op = tf.group(d_in_train_op, train_op)
    train_op = tf.group(g_train_op, d_train_op, d_in_train_op, train_op)
    total_loss = tf.losses.get_total_loss()
    with tf.control_dependencies([train_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

#    # summary op
#    summary_op = tf.summary.merge_all()
#
#    with tf.Session(config=session_config) as sess:
#      sess.run(tf.global_variables_initializer())
#      summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
#      #optimistic_restore(sess, train_config.fine_tune_checkpoint)
#
#      # Start the queue runners.
#      tf.train.start_queue_runners(sess=sess)
#
#      for iter in xrange(0, train_config.num_steps): 
#        _start_time = time.time()
#        if g_train_op is not None:
#          _ = sess.run(g_train_op)
#        if d_train_op is not None:
#          _ = sess.run(d_train_op)
#        if d_in_train_op is not None:
#          _ = sess.run(d_in_train_op)
#        if train_op is not None:
#          global_step_val, _ = sess.run([global_step, train_op])
#
#        duration = time.time() - _start_time
#        examples_per_sec = train_config.batch_size / duration
#        sec_per_batch = float(duration)
#
#        format_str = ('%s: step %d  '
#                      ', (%.1f examples/sec; %.3f sec/batch)')
#        print (format_str % (datetime.now(), global_step_val,
#                             examples_per_sec, sec_per_batch))
#
#        if (iter+1) % 100 == 0:
#          summary = sess.run(summary_op)
#          summary_writer.add_summary(summary, iter+1)
#
#        if (iter+1) % 2000 == 0:
#          saver.save(sess, FLAGS.train_dir + '/model.ckpt', global_step=global_step)
#
#      saver.save(sess, FLAGS.train_dir + '/model.ckpt', global_step=global_step)

#    # Gather update_ops from the first clone. These contain, for example,
#    # the updates for the batch_norm variables created by model_fn.
#    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
#
    with tf.device(deploy_config.optimizer_device()):
      training_optimizer = optimizer_builder.build(train_config.optimizer,
                                                   global_summaries)

    sync_optimizer = None
    if train_config.sync_replicas:
      training_optimizer = tf.SyncReplicasOptimizer(
          training_optimizer,
          replicas_to_aggregate=train_config.replicas_to_aggregate,
          total_num_replicas=train_config.worker_replicas)
      sync_optimizer = training_optimizer

    # Create ops required to initialize the model from a given checkpoint.
    init_fn = None
    if train_config.fine_tune_checkpoint:
      var_map = detection_model.restore_map(
          from_detection_checkpoint=train_config.from_detection_checkpoint)
      available_var_map = (variables_helper.
                           get_variables_available_in_checkpoint(
                               var_map, train_config.fine_tune_checkpoint))
      init_saver = tf.train.Saver(available_var_map)
      def initializer_fn(sess):
        print('init')
        init_saver.restore(sess, train_config.fine_tune_checkpoint)
      init_fn = initializer_fn

    with tf.device(deploy_config.optimizer_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, training_optimizer, regularization_losses=None)
      total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

      # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
      if train_config.bias_grad_multiplier:
        biases_regex_list = ['.*/biases']
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(
            grads_and_vars,
            biases_regex_list,
            multiplier=train_config.bias_grad_multiplier)

      # Optionally freeze some layers by setting their gradients to be zero.
      if train_config.freeze_variables:
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(
            grads_and_vars, train_config.freeze_variables)

      # Optionally clip gradients
      if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
          grads_and_vars = slim.learning.clip_gradient_norms(
              grads_and_vars, train_config.gradient_clipping_by_norm)

      # Create gradient updates.
      grad_updates = training_optimizer.apply_gradients(grads_and_vars,
                                                        global_step=global_step)
#      update_ops.append(grad_updates)
#
#      update_op = tf.group(*update_ops)
#      with tf.control_dependencies([update_op]):
#        train_tensor = tf.identity(total_loss, name='train_op')

    # Add summaries.
    for model_var in slim.get_model_variables():
      global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
    for loss_tensor in tf.losses.get_losses():
      global_summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
    global_summaries.add(
        tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    summaries |= global_summaries

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    # Save checkpoints regularly.
    keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
    saver = tf.train.Saver(
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)


    slim.learning.train(
        train_tensor,
        logdir=train_dir,
        master=master,
        is_chief=is_chief,
        session_config=session_config,
        startup_delay_steps=train_config.startup_delay_steps,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=(
            train_config.num_steps if train_config.num_steps else None),
        save_summaries_secs=120,
        #sync_optimizer=sync_optimizer,
        saver=saver)
