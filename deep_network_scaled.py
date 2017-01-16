# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import ops

print('injecting methods from ops into sugartensor')
tf.sg_inject_func(ops.sg_conv_with_scale)
tf.sg_inject_func(ops.sg_upconv_with_scale)

# set log level to debug
tf.sg_verbosity(10)


batch_size = 4   # batch size
z_dim = 50        # noise dimension
margin = 1        # max-margin for hinge loss
pt_weight = 0.1  # PT regularizer's weight


INPUT_SIZE = 50

z = tf.random_uniform((batch_size, INPUT_SIZE, INPUT_SIZE, 1))


with tf.sg_context(name='deep_scaled', size=7, stride=1, act='relu', bn=False, batch_size=batch_size):
  out = (z.sg_conv_with_scale(dim=1)
      .sg_conv_with_scale(dim=1)
      .sg_conv_with_scale(dim=1)
      .sg_conv_with_scale(dim=1)
      .sg_conv_with_scale(dim=1)
      .sg_conv_with_scale(dim=1, act='sigmoid')
      .sg_squeeze())


with tf.Session() as sess:
  tf.sg_init(sess)
  imgs = sess.run(out)

  ops.plot_images(imgs)


