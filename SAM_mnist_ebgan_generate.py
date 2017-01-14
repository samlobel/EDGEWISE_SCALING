# -*- coding: utf-8 -*-
import sugartensor as tf
import matplotlib.pyplot as plt
import ops

print('injecting methods from ops into sugartensor')
tf.sg_inject_func(ops.sg_conv_with_scale)
tf.sg_inject_func(ops.sg_upconv_with_scale)

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 25
z_dim = 50


#
# create generator
#

# random uniform seed
z = tf.random_uniform((batch_size, z_dim))

with tf.sg_context(name='generator', size=5, stride=2, act='relu', bn=True, batch_size=batch_size):
    # generator network
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=7*7*128)
           .sg_reshape(shape=(-1, 7, 7, 128))
           .sg_upconv_with_scale(dim=64)
           .sg_upconv_with_scale(dim=1, act='sigmoid', bn=False)
           .sg_squeeze())


#
# draw samples
#

with tf.Session() as sess:
    tf.sg_init(sess)
    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))

    # run generator
    imgs = sess.run(gen)

    ops.plot_images(imgs)
    # plot result
    # _, ax = plt.subplots(10, 10, sharex=True, sharey=True)
    # for i in range(10):
    #     for j in range(10):
    #         ax[i][j].imshow(imgs[i * 10 + j], 'gray')
    #         ax[i][j].set_axis_off()
    # plt.savefig('asset/train/sample.png', dpi=600)
    # tf.sg_info('Sample image saved to "asset/train/sample.png"')
    plt.close()
