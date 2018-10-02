
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#if tf.__version__ != '1.4.0':
#  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')


# ## Env setup

# In[2]:


# This is needed to display the images.
#get_ipython().magic(u'matplotlib inline')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:


# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_08'
#MODEL_NAME = 'inference_graph/rfcn_resnet101_coco_pivot'
MODEL_NAME = 'inference_graph/faster_rcnn_inception_v2_coco'
#MODEL_NAME = 'inference_graph/faster_rcnn_inception_v2_coco_gaussian_lr_avgfilter_discrim'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[5]:


#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Name of intermediate nodes for faster_rcnn_inception_v2
intermediate_nodes = ['FirstStageFeatureExtractor/InceptionV2/InceptionV2/Conv2d_1a_7x7/BatchNorm/FusedBatchNorm:0',
                      'FirstStageFeatureExtractor/InceptionV2/InceptionV2/Conv2d_2b_1x1/BatchNorm/FusedBatchNorm:0',
                      'FirstStageFeatureExtractor/InceptionV2/InceptionV2/Conv2d_2c_3x3/BatchNorm/FusedBatchNorm:0',
                      'FirstStageFeatureExtractor/InceptionV2/InceptionV2/Mixed_3b/concat:0',
                      'FirstStageFeatureExtractor/InceptionV2/InceptionV2/Mixed_3c/concat:0',
                      'FirstStageFeatureExtractor/InceptionV2/InceptionV2/Mixed_4a/concat:0',
                      'FirstStageFeatureExtractor/InceptionV2/InceptionV2/Mixed_4b/concat:0',
                      'FirstStageFeatureExtractor/InceptionV2/InceptionV2/Mixed_4c/concat:0',
                      'FirstStageFeatureExtractor/InceptionV2/InceptionV2/Mixed_4d/concat:0',
                      'FirstStageFeatureExtractor/InceptionV2/InceptionV2/Mixed_4e/concat:0',
                      'SecondStageFeatureExtractor/InceptionV2/Mixed_5a/concat:0',
                      'SecondStageFeatureExtractor/InceptionV2/Mixed_5b/concat:0',
                      'SecondStageFeatureExtractor/InceptionV2/Mixed_5c/concat:0',
                     ]


# In[10]:

def add_gaussian_noise(image_np, std=0.1):
  # input
  # image_np: (?,?,3) in range [0, 255], dtype=uint8
  noisy_image_np = np.copy(image_np)
  noisy_image_np = noisy_image_np.astype('float32')
  # generate noise
  gaussian_noise = np.random.normal(0, std*255, size=image_np.shape)
  noisy_image_np += gaussian_noise
  noisy_image_np = np.clip(noisy_image_np+0.5, 0., 255.)
  # essentially floor operation
  noisy_image_np = noisy_image_np.astype('uint8')

  return noisy_image_np


def normalize(inter, noisy_inter):
  mean = np.mean(inter)
  var = np.var(inter)
  inter = (inter-mean)/var**0.5
  noisy_inter = (noisy_inter-mean)/var**0.5
  return inter, noisy_inter

with detection_graph.as_default():
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  with tf.Session(graph=detection_graph, config=run_config) as sess:
    # Print all tensors in detection_graph
    #for name in [n.name for n in tf.get_default_graph().as_graph_def().node]:
    #for n in [n for n in tf.get_default_graph().as_graph_def().node]:
    #  print (n.name, n)
    for i in tf.get_default_graph().get_operations():
      if i.name in intermediate_nodes:
        print (i.name, i.values())
    # Get intermediate nodes
    intermediate_tensors = []
    for node in intermediate_nodes:
      intermediate_tensors.append(detection_graph.get_tensor_by_name(node))
    
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      noisy_image_np = add_gaussian_noise(image_np)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      noisy_image_np_expanded = np.expand_dims(noisy_image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: noisy_image_np_expanded})
      (noisy_boxes, noisy_scores, noisy_classes, noisy_num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      intermediates = sess.run(intermediate_tensors,
          feed_dict={image_tensor: image_np_expanded})
      noisy_intermediates = sess.run(intermediate_tensors,
          feed_dict={image_tensor: noisy_image_np_expanded})

      # Normalize
      intermediates_normalized = []
      noisy_intermediates_normalized = []
      mses = []
      for inter, noisy_inter in zip(intermediates, noisy_intermediates):
        intermediate_normalized, noisy_intermediate_normalized = normalize(inter, noisy_inter)
        intermediates_normalized.append(intermediate_normalized)
        noisy_intermediates_normalized.append(noisy_intermediate_normalized)
        mses.append(np.mean((intermediate_normalized-noisy_intermediate_normalized)**2))
      
      input_normalized, noisy_input_normalized = normalize(image_np_expanded, noisy_image_np_expanded)
      input_mse = np.mean((input_normalized-noisy_input_normalized)**2)
      print ('input', input_mse)
      for node, mse in zip(intermediate_nodes, mses):
        print (node, mse)

      # Difference

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      vis_util.visualize_boxes_and_labels_on_image_array(
          noisy_image_np,
          np.squeeze(noisy_boxes),
          np.squeeze(noisy_classes).astype(np.int32),
          np.squeeze(noisy_scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      plt.show(block=False)
      plt.savefig(image_path.replace('.jpg', '_detection.jpg'))
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(noisy_image_np)
      plt.show(block=False)
      plt.savefig(image_path.replace('.jpg', '_noisy_detection.jpg'))

      LINE_WIDTH = 4
      fig, ax = plt.subplots()
      ax.plot(range(len(mses)+1), [input_mse] + mses, label='MSE', linewidth=LINE_WIDTH)
      plt.grid()
      plt.xlabel('layer index')
      plt.ylabel('Normalized MSE')
      plt.show(block=False)
      plt.savefig(image_path.replace('.jpg', '_mse.png'))


# In[ ]:





# In[ ]:




