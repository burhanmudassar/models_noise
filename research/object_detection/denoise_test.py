import tensorflow as tf
import numpy as np
# input image with 10x10 shape for 3 channels
# filter with 10x10 shape for each input channel

N_in_channel = 3
N_out_channel_mul = 8
x = tf.random_normal([2, 10, 10, N_in_channel])
f = tf.random_normal([3, 3, N_in_channel, N_out_channel_mul])
fp = tf.random_normal([1, 1, N_in_channel*N_out_channel_mul, 20])
y = tf.nn.depthwise_conv2d(x, f, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
y2 = tf.nn.separable_conv2d(x, depthwise_filter=f,
                            pointwise_filter=fp, 
                            strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
x_concat = tf.concat(x, 0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_data, x_c_data, f_data, y_conv, y_conv2 = sess.run([x, x_concat, f, y, y2])

print(x_data.shape)
print(x_c_data.shape)
print(y_conv.shape)
print(y_conv2.shape)

sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_data, x_c_data, f_data, y_conv, y_conv2 = sess.run([x, x_concat, f, y, y2])

print(x_data.shape)
print(x_c_data.shape)
print(y_conv.shape)
print(y_conv2.shape)


#y_s = np.squeeze(y_conv)
#for i in range(N_in_channel):
#    for j in range(N_out_channel_mul):
#        print("np: %f, tf:%f" % (np.sum(x_data[0, :, :, i] * f_data[:, :, i, j]), y_s[i * N_out_channel_mul + j]))
