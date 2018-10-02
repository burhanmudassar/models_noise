
import tensorflow as tf
import numpy as np

image = 1.0*tf.ones([1, 30, 40, 3])
max_e = 0.15


image_shape = tf.shape(image)
num_batch = image_shape[0]
height = image_shape[1]
width = image_shape[2]

sigma = tf.abs(tf.truncated_normal([num_batch], mean=0., stddev=max_e/2))
noise = tf.random_normal(image_shape, mean=0., stddev=sigma)
noise_image = tf.clip_by_value(image + noise, 0, 1.)



sess = tf.Session()
try:
  tf.global_variables_initializer().run(session=sess)
except:
  tf.initialize_all_variables().run(session=sess)
#sess.run(snow_rows)
print(sess.run(image))
print(sess.run(noise_image))
for i in xrange(100):
  print(sess.run(sigma))
