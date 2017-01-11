import sugartensor as tf
import numpy as np
# init = tf.sg_initializer
# sg_act = tf.sg_activation
from sugartensor import sg_initializer as init
from sugartensor import sg_activation as sg_act
import matplotlib.pyplot as plt
import os

# def get_sg_act(act_name):
#   return getattr(sg_act, 'sg_' + act_name.lower())


def make_scaling_matrix_for_conv(input_shape, filter_shape, strides, padding='SAME'):
  INPUT_ONES = np.ones(input_shape, dtype=np.float32)
  FILTER_ONES = np.ones(filter_shape, dtype=np.float32)
  output = tf.nn.conv2d(INPUT_ONES, FILTER_ONES, strides=strides, padding=padding)
  output = tf.div(output, tf.reduce_mean(output))
  output = tf.div(1.0, output)
  return output
  # max_output = tf.reduce_max(output)
  # norm_output = tf.div(output, max_output)
  # inv_norm_output = tf.div(1.0, norm_output)
  # return inv_norm_output

def make_scaling_matrix_for_conv_transpose(input_shape, filter_shape, output_shape, strides, padding='SAME'):
  INPUT_ONES = np.ones(input_shape, dtype=np.float32)
  FILTER_ONES = np.ones(filter_shape, dtype=np.float32)
  output = tf.nn.conv2d_transpose(INPUT_ONES, FILTER_ONES, output_shape=output_shape, strides=strides, padding=padding)
  # max_output = tf.reduce_max(output)
  # norm_output = tf.div(output, max_output)
  # inv_norm_output = tf.div(1.0, norm_output)
  # return inv_norm_output
  output = tf.div(output, tf.reduce_mean(output))
  output = tf.div(1.0, output)
  return output  


@tf.sg_layer_func
def sg_conv_with_scale(tensor, opt):
    opt += tf.sg_opt(size=(3, 3), stride=(1, 1, 1, 1), pad='SAME')
    opt.size = opt.size if isinstance(opt.size, (tuple, list)) else [opt.size, opt.size]
    opt.stride = opt.stride if isinstance(opt.stride, (tuple, list)) else [1, opt.stride, opt.stride, 1]
    opt.stride = [1, opt.stride[0], opt.stride[1], 1] if len(opt.stride) == 2 else opt.stride

    # parameter initialize
    filter_shape = (opt.size[0], opt.size[1], opt.in_dim, opt.dim)
    w = tf.sg_initializer.he_uniform('W', filter_shape)
    b = tf.sg_initializer.constant('b', opt.dim) if opt.bias else 0

    # apply convolution
    conv_unscaled = tf.nn.conv2d(tensor, w, strides=opt.stride, padding=opt.pad)# + b

    # input_shape = [int(d) for d in tensor.get_shape()]
    shape = tensor.get_shape().as_list()    
    input_shape = [1, shape[1],shape[2], shape[3]]

    # Make scaler
    scaler=make_scaling_matrix_for_conv(input_shape, filter_shape, opt.stride)
    # Apply scaler and bias
    conv_scaled = tf.mul(conv_unscaled, scaler) + b

    return conv_scaled

@tf.sg_layer_func
def sg_upconv_with_scale(tensor, opt):
    opt += tf.sg_opt(size=(4, 4), stride=(1, 2, 2, 1), pad='SAME')
    opt.size = opt.size if isinstance(opt.size, (tuple, list)) else [opt.size, opt.size]
    opt.stride = opt.stride if isinstance(opt.stride, (tuple, list)) else [1, opt.stride, opt.stride, 1]
    opt.stride = [1, opt.stride[0], opt.stride[1], 1] if len(opt.stride) == 2 else opt.stride

    # parameter tf.sg_initializer
    tensor_shape = tensor.get_shape().as_list()
    filter_shape = (opt.size[0], opt.size[1], opt.dim, opt.in_dim)
    # print('FILTER_SHAPE: {}       TENSOR_SHAPE: {}'.format(filter_shape, tensor_shape))
    w = tf.sg_initializer.he_uniform('W', filter_shape)
    b = tf.sg_initializer.constant('b', opt.dim) if opt.bias else 0

    # tedious shape handling for conv2d_transpose
    shape = tensor.get_shape().as_list()
    out_shape = [opt.batch_size, shape[1] * opt.stride[1], shape[2] * opt.stride[2], opt.dim]

    # apply convolution
    out_unscaled = tf.nn.conv2d_transpose(tensor, w, output_shape=tf.pack(out_shape),
                                 strides=opt.stride, padding=opt.pad)# + b
    # reset shape is needed because conv2d_transpose() erase all shape information.
    # noinspection PyUnresolvedReferences
    out_unscaled.set_shape(out_shape)

    # out_unscaled_shape = out_unscaled.get_shape().as_list()
    # print('out_unscaled_shape: {}'.format(out_unscaled_shape))

    # return out_unscaled

    # print(tensor.get_shape())
    # print([type(val) for val in tensor.get_shape()])

    # input_shape = [int(d) for d in tensor.get_shape()]
    # print(input_shape)
    input_shape = [opt.batch_size, shape[1],shape[2], shape[3]]
    # print(input_shape)
    scaler=make_scaling_matrix_for_conv_transpose(input_shape, filter_shape, out_shape, opt.stride)

    # final_out_shape = [opt.batch_size, shape[1] * opt.stride[1], shape[2] * opt.stride[2], opt.dim]
    # print('final out shape: {}'.format(final_out_shape))
    # print(final_out_shape)

    out_scaled = tf.mul(out_unscaled, scaler) + b
    # out_scaled_shape = out_scaled.get_shape().as_list()
    # print('out_scaled_shape: {}'.format(out_scaled_shape))
    # out_scaled.set_shape([tensor.get_shape()[0], out_shape[1], out_shape[2], opt.dim])
    out_scaled.set_shape(out_shape)
    # out_scaled.set_shape([None, out_shape[1], out_shape[2], opt.dim])
    # print('\n')
    return out_scaled

"""
************************ BEGIN OLD  ******************************************************
"""


# def make_scaling_matrix_for_conv(input_shape, filter_shape, strides, padding='SAME'):
#   INPUT_ONES = np.ones(input_shape, dtype=np.float32)
#   FILTER_ONES = np.ones(filter_shape, dtype=np.float32)
#   output = tf.nn.conv2d(INPUT_ONES, FILTER_ONES, strides=strides, padding=padding)
#   max_output = tf.reduce_max(output)
#   norm_output = tf.div(output, max_output)
#   inv_norm_output = tf.div(1.0, norm_output)
#   return inv_norm_output

# def make_scaling_matrix_for_conv_transpose(input_shape, filter_shape, output_shape, strides, padding='SAME'):
#   INPUT_ONES = np.ones(input_shape, dtype=np.float32)
#   FILTER_ONES = np.ones(filter_shape, dtype=np.float32)
#   output = tf.nn.conv2d_transpose(INPUT_ONES, FILTER_ONES, output_shape=output_shape, strides=strides, padding=padding)
#   max_output = tf.reduce_max(output)
#   norm_output = tf.div(output, max_output)
#   inv_norm_output = tf.div(1.0, norm_output)
#   return inv_norm_output


# Why don't I want to do my own conv? It's mainly for logging purposes...
# Because his variables log and initialize well. I really don't want to give that up.

# def conv_and_scale(tensor, dim, size, stride, act, bn=False, bias=None):
#   # size is Kernal Size.
#   # I wonder if opt passes down. Anyways...
#   in_shape = [int(d) for d in tensor.get_shape()]
#   filter_shape = [size, size, in_shape[-1], dim]
#   strides = [1, stride, stride, 1]

#   conv = tensor.sg_conv(size=size, dim=dim, stride=stride, act='linear', bn=False, bias=False) #linear at first
#   scaler = make_scaling_matrix_for_conv(in_shape, filter_shape, strides)
#   scaled_conv = tf.mul(conv, scaler)
#   scaled_with_options = scaled_conv.sg_bypass(act=act, bn=bn, bias=bias)
#   return scaled_with_options

#   # if bn:
#   #   b = init.constant('b', dim)
#   # scaled_conv = scaled_conv + (b if bn else 0)

#   # act_fun = get_sg_act(act)
#   # print('act_fun: {}'.format(act_fun))
#   # out = act_fun(scaled_conv)
#   # return out


# def upconv_and_scale(tensor, dim, size, stride, act, bn=False, bias=None):
#   # size is Kernal Size.
#   # I wonder if opt passes down. Anyways...
#   in_shape = [int(d) for d in tensor.get_shape()]
#   print('in_shape for upconv: {}'.format(in_shape))
#   filter_shape = [size, size, dim, in_shape[-1]]
#   print('filter_shape for upconv: {}'.format(filter_shape))
#   out_shape = [in_shape[0], in_shape[1]*stride, in_shape[2]*stride, dim]
#   print('out_shape for upconv: {}'.format(out_shape))
#   strides = [1, stride, stride, 1]

#   upconv = tensor.sg_upconv(size=size, dim=dim, stride=stride, act='linear', bn=bn, bias=False) #linear at first
#   # Actually, batch normalization makes no difference here... Because it works on channels and not pixels
#   print('upconv_shape is {}'.format(upconv.get_shape()))
#   scaler = make_scaling_matrix_for_conv_transpose(in_shape, filter_shape, out_shape, strides)
#   print('scaler_shape is {}'.format(scaler.get_shape()))
#   scaled_upconv = tf.mul(upconv, scaler)
#   scaled_with_options = scaled_upconv.sg_bypass(act=act, bn=bn, bias=bias)
#   return scaled_with_options

  # if bn:
  #   b = init.constant('b', dim)
  # scaled_upconv = scaled_upconv + (b if bn else 0)

  # act_fun = get_sg_act(act)
  # print('act_fun: {}'.format(act_fun))
  # out = act_fun(scaled_upconv) #BN HAPPENS HERE.
  # return out


"""
************************ END OLD  ******************************************************
"""


asset_folder = 'asset/train/'
def get_next_filename():
  i = 0
  while True:
    num_str = str(i).zfill(4)
    filename = 'sample{}.png'.format(num_str)
    filename = os.path.join(asset_folder, filename)
    if not os.path.isfile(filename):
      print('next filename: {}'.format(filename))
      return filename
    i += 1


def plot_images(imgs, filename=None):
  # plot result
  num_images = imgs.shape[0]
  num_per_square = int(num_images**0.5)
  # print("num per square: " + str(num_per_square))

  _, ax = plt.subplots(num_per_square, num_per_square, sharex=True, sharey=True)
  for i in range(num_per_square):
      for j in range(num_per_square):
          ax[i][j].imshow(imgs[i * num_per_square + j], 'gray')
          ax[i][j].set_axis_off()
  filename=filename or get_next_filename()
  plt.savefig(filename, dpi=200)
  tf.sg_info('Sample image saved to "{}"'.format(filename))
  plt.close()




  


if __name__ == '__main__':
  print('testing')
  print('testing on matrix: ')
  with tf.Session() as sess:
    input_shape = [1,8,8,1]
    filter_shape = [3,3,1,1]
    strides = [1,2,2,1]
    mat = make_scaling_matrix_for_conv(input_shape,filter_shape,strides)
    print('tensor for inputs: {}, {}, {}'.format(input_shape,filter_shape,strides))
    print(mat.eval())
  print('exited')


  # http://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers is soooo great.
  # Also, https://github.com/vdumoulin/conv_arithmetic is the source.



