
import tensorflow as tf
import numpy as np

image = 0.5*tf.ones([1, 30, 40, 3])
sparsity = 0.5




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
snow_ball0 = tf.constant([0.,0.,1.,1.,0.,0.])
snow_ball1 = tf.constant([0.,1.,1.,1.,1.,0.])
snow_ball2 = tf.constant([1.,1.,1.,1.,1.,1.])
snow_ball_upper = tf.stack([snow_ball0, snow_ball1, snow_ball2])
snow_ball_lower = tf.stack([snow_ball2, snow_ball1, snow_ball0])
snow_ball = tf.concat([snow_ball_upper, snow_ball_lower], 0)

num_snows = int(height*width*sparsity/24)

#snow_rows = tf.random_uniform([num_snows], 0, height-6, dtype=tf.int32)
#snow_cols = tf.random_uniform([num_snows], 0, width-6, dtype=tf.int32)

snow_rows = np.random.randint(width-6, size=num_snows)
snow_cols = np.random.randint(height-6, size=num_snows)

mask = tf.get_variable('snow_mask', image_shape, dtype=tf.float32,
                       initializer=tf.constant_initializer(0), trainable=False)
mask.assign(tf.zeros_like(image))
for (row, col) in zip(snow_rows, snow_cols):
  mask[:, row:row+6, col:col+6, :].assign(snow_ball)




sess = tf.Session()
try:
  tf.global_variables_initializer().run(session=sess)
except:
  tf.initialize_all_variables().run(session=sess)
#sess.run(snow_rows)
sess.run(snow_ball)
print(sess.run(mask))
