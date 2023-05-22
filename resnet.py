# coding=utf-8
# Copyright 2020 The SimCLR Authors.
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
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the post-activation form of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow.compat.v1 as tf

from tensorflow.python.tpu import tpu_function  # pylint: disable=g-direct-tensorflow-import


FLAGS = flags.FLAGS
BATCH_NORM_EPSILON = 1e-5


class BatchNormalization(tf.layers.BatchNormalization):
  """Batch Normalization layer that supports cross replica computation on TPU.

  This class extends the keras.BatchNormalization implementation by supporting
  cross replica means and variances. The base class implementation only computes
  moments based on mini-batch per replica (TPU core).

  For detailed information of arguments and implementation, refer to:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
  """

  def __init__(self, fused=False, **kwargs):
    """Builds the batch normalization layer.

    Arguments:
      fused: If `False`, use the system recommended implementation. Only support
        `False` in the current implementation.
      **kwargs: input augments that are forwarded to
        tf.layers.BatchNormalization.
    """
    if fused in (True, None):
      raise ValueError('The TPU version of BatchNormalization does not support '
                       'fused=True.')
    super(BatchNormalization, self).__init__(fused=fused, **kwargs)

  def _cross_replica_average(self, t):
    """Calculates the average value of input tensor across TPU replicas."""
    num_shards = tpu_function.get_tpu_context().number_of_shards
    return tf.tpu.cross_replica_sum(t) / tf.cast(num_shards, t.dtype)

  def _moments(self, inputs, reduction_axes, keep_dims, mask=None):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super(BatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims, mask=mask)

    num_shards = tpu_function.get_tpu_context().number_of_shards
    if num_shards and num_shards > 1:
      # Each group has multiple replicas: here we compute group mean/variance by
      # aggregating per-replica mean/variance.
      group_mean = self._cross_replica_average(shard_mean)
      group_variance = self._cross_replica_average(shard_variance)

      # Group variance needs to also include the difference between shard_mean
      # and group_mean.
      mean_distance = tf.square(group_mean - shard_mean)
      group_variance += self._cross_replica_average(mean_distance)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)


def batch_norm_relu(inputs, is_training, relu=True, init_zero=False,
                    center=True, scale=True, data_format='channels_last'):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    center: `bool` whether to add learnable bias factor.
    scale: `bool` whether to add learnable scaling factor.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = -1

  if FLAGS.global_bn:
    bn_foo = BatchNormalization(
        axis=axis,
        momentum=FLAGS.batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON,
        center=center,
        scale=scale,
        fused=False,
        gamma_initializer=gamma_initializer)
    inputs = bn_foo(inputs, training=is_training)
  else:
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=FLAGS.batch_norm_decay,
        epsilon=BATCH_NORM_EPSILON,
        center=center,
        scale=scale,
        training=is_training,
        fused=True,
        gamma_initializer=gamma_initializer)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def dropblock(net, is_training, keep_prob, dropblock_size,
              data_format='channels_last'):
  """DropBlock: a regularization method for convolutional neural networks.

  DropBlock is a form of structured dropout, where units in a contiguous
  region of a feature map are dropped together. DropBlock works better than
  dropout on convolutional layers due to the fact that activation units in
  convolutional layers are spatially correlated.
  See https://arxiv.org/pdf/1810.12890.pdf for details.

  Args:
    net: `Tensor` input tensor.
    is_training: `bool` for whether the model is training.
    keep_prob: `float` or `Tensor` keep_prob parameter of DropBlock. "None"
        means no DropBlock.
    dropblock_size: `int` size of blocks to be dropped by DropBlock.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
  Returns:
      A version of input tensor with DropBlock applied.
  Raises:
      if width and height of the input tensor are not equal.
  """

  if not is_training or keep_prob is None:
    return net

  tf.logging.info('Applying DropBlock: dropblock_size {}, net.shape {}'.format(
      dropblock_size, net.shape))

  if data_format == 'channels_last':
    _, width, height, _ = net.get_shape().as_list()
  else:
    _, _, width, height = net.get_shape().as_list()
  if width != height:
    raise ValueError('Input tensor with width!=height is not supported.')

  dropblock_size = min(dropblock_size, width)
  # seed_drop_rate is the gamma parameter of DropBlcok.
  seed_drop_rate = (1.0 - keep_prob) * width**2 / dropblock_size**2 / (
      width - dropblock_size + 1)**2

  # Forces the block to be inside the feature map.
  w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
  valid_block_center = tf.logical_and(
      tf.logical_and(w_i >= int(dropblock_size // 2),
                     w_i < width - (dropblock_size - 1) // 2),
      tf.logical_and(h_i >= int(dropblock_size // 2),
                     h_i < width - (dropblock_size - 1) // 2))

  valid_block_center = tf.expand_dims(valid_block_center, 0)
  valid_block_center = tf.expand_dims(
      valid_block_center, -1 if data_format == 'channels_last' else 0)

  randnoise = tf.random_uniform(net.shape, dtype=tf.float32)
  block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
      (1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1
  block_pattern = tf.cast(block_pattern, dtype=tf.float32)

  if dropblock_size == width:
    block_pattern = tf.reduce_min(
        block_pattern,
        axis=[1, 2] if data_format == 'channels_last' else [2, 3],
        keepdims=True)
  else:
    if data_format == 'channels_last':
      ksize = [1, dropblock_size, dropblock_size, 1]
    else:
      ksize = [1, 1, dropblock_size, dropblock_size]
    block_pattern = -tf.nn.max_pool(
        -block_pattern, ksize=ksize, strides=[1, 1, 1, 1], padding='SAME',
        data_format='NHWC' if data_format == 'channels_last' else 'NCHW')

  percent_ones = tf.cast(tf.reduce_sum((block_pattern)), tf.float32) / tf.cast(
      tf.size(block_pattern), tf.float32)

  net = net / tf.cast(percent_ones, net.dtype) * tf.cast(
      block_pattern, net.dtype)
  return net


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides,
                         data_format='channels_last'):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def sk_conv2d(inputs, filters, strides, sk_ratio, min_dim=32,
              is_training=True, data_format='channels_last'):
  """Selective kernel convolutional layer (https://arxiv.org/abs/1903.06586)."""
  channel_axis = 1 if data_format == 'channels_first' else 3
  pooling_axes = [2, 3] if data_format == 'channels_first' else [1, 2]

  # Two stream convs (using split and both are 3x3).
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=2 * filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
  inputs = tf.stack(tf.split(inputs, num_or_size_splits=2, axis=channel_axis))

  # Mixing weights for two streams.
  mid_dim = max(int(filters * sk_ratio), min_dim)
  global_features = tf.reduce_mean(
      tf.reduce_sum(inputs, axis=0), pooling_axes, keepdims=True)
  global_features = tf.layers.conv2d(
      inputs=global_features, filters=mid_dim, kernel_size=1, strides=1,
      kernel_initializer=tf.variance_scaling_initializer(),
      use_bias=False, data_format=data_format)
  global_features = batch_norm_relu(
      global_features, is_training, data_format=data_format)
  mixing = tf.layers.conv2d(
      inputs=global_features, filters=2 * filters, kernel_size=1, strides=1,
      kernel_initializer=tf.variance_scaling_initializer(),
      use_bias=False, data_format=data_format)
  mixing = tf.stack(tf.split(mixing, num_or_size_splits=2, axis=channel_axis))
  mixing = tf.nn.softmax(mixing, axis=0)

  return tf.reduce_sum(inputs * mixing, axis=0)


def se_layer(inputs, filters, se_ratio, data_format='channels_last'):
  """Squeeze and Excitation layer (https://arxiv.org/abs/1709.01507)."""
  if se_ratio <= 0:
    return inputs
  se_reduce = tf.layers.Conv2D(
      max(1, int(filters * se_ratio)),
      kernel_size=[1, 1],
      strides=[1, 1],
      kernel_initializer=tf.variance_scaling_initializer(),
      padding='same',
      data_format=data_format,
      use_bias=True)
  se_expand = tf.layers.Conv2D(
      inputs.shape[-1],
      kernel_size=[1, 1],
      strides=[1, 1],
      kernel_initializer=tf.variance_scaling_initializer(),
      padding='same',
      data_format=data_format,
      use_bias=True)

  spatial_dims = [2, 3] if data_format == 'channels_first' else [1, 2]
  se_tensor = tf.reduce_mean(
      inputs, spatial_dims, keepdims=True)
  se_tensor = se_expand(tf.nn.relu(se_reduce(se_tensor)))
  return tf.sigmoid(se_tensor) * inputs


def residual_block(inputs, filters, is_training, strides,
                   use_projection=False, data_format='channels_last',
                   dropblock_keep_prob=None, dropblock_size=None):
  """Standard building block for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    dropblock_keep_prob: unused; needed to give method same signature as other
      blocks
    dropblock_size: unused; needed to give method same signature as other
      blocks
  Returns:
    The output `Tensor` of the block.
  """
  del dropblock_keep_prob
  del dropblock_size
  shortcut = inputs
  if use_projection:
    # Projection shortcut in first layer to match filters and strides
    if FLAGS.sk_ratio > 0:  # Use ResNet-D (https://arxiv.org/abs/1812.01187)
      if strides > 1:
        inputs = fixed_padding(inputs, 2, data_format)
      inputs = tf.layers.average_pooling2d(
          inputs, pool_size=2, strides=strides,
          padding='SAME' if strides == 1 else 'VALID', data_format=data_format)
      shortcut = conv2d_fixed_padding(
          inputs=inputs, filters=filters, kernel_size=1, strides=1,
          data_format=data_format)
    else:
      shortcut = conv2d_fixed_padding(
          inputs=inputs, filters=filters, kernel_size=1, strides=strides,
          data_format=data_format)
    shortcut = batch_norm_relu(shortcut, is_training, relu=False,
                               data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, relu=False, init_zero=True,
                           data_format=data_format)

  if FLAGS.se_ratio > 0:
    inputs = se_layer(inputs, filters, FLAGS.se_ratio, data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs, filters, is_training, strides,
                     use_projection=False, data_format='channels_last',
                     dropblock_keep_prob=None, dropblock_size=None):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    dropblock_keep_prob: `float` or `Tensor` keep_prob parameter of DropBlock.
        "None" means no DropBlock.
    dropblock_size: `int` size parameter of DropBlock. Will not be used if
        dropblock_keep_prob is "None".

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    if FLAGS.sk_ratio > 0:  # Use ResNet-D (https://arxiv.org/abs/1812.01187)
      if strides > 1:
        shortcut = fixed_padding(inputs, 2, data_format)
      else:
        shortcut = inputs
      shortcut = tf.layers.average_pooling2d(
          shortcut, pool_size=2, strides=strides,
          padding='SAME' if strides == 1 else 'VALID', data_format=data_format)
      shortcut = conv2d_fixed_padding(
          inputs=shortcut, filters=filters_out, kernel_size=1, strides=1,
          data_format=data_format)
    else:
      shortcut = conv2d_fixed_padding(
          inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
          data_format=data_format)
    shortcut = batch_norm_relu(shortcut, is_training, relu=False,
                               data_format=data_format)
  shortcut = dropblock(
      shortcut, is_training=is_training, data_format=data_format,
      keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
  inputs = dropblock(
      inputs, is_training=is_training, data_format=data_format,
      keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size)

  if FLAGS.sk_ratio > 0:
    inputs = sk_conv2d(
        inputs, filters, strides, FLAGS.sk_ratio,
        is_training=is_training, data_format=data_format)
  else:
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)
    inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
  inputs = dropblock(
      inputs, is_training=is_training, data_format=data_format,
      keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, relu=False, init_zero=True,
                           data_format=data_format)
  inputs = dropblock(
      inputs, is_training=is_training, data_format=data_format,
      keep_prob=dropblock_keep_prob, dropblock_size=dropblock_size)

  if FLAGS.se_ratio > 0:
    inputs = se_layer(inputs, filters, FLAGS.se_ratio, data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def block_group(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format='channels_last', dropblock_keep_prob=None,
                dropblock_size=None):
  """Creates one group of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
    is_training: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    dropblock_keep_prob: `float` or `Tensor` keep_prob parameter of DropBlock.
        "None" means no DropBlock.
    dropblock_size: `int` size parameter of DropBlock. Will not be used if
        dropblock_keep_prob is "None".

  Returns:
    The output `Tensor` of the block layer.
  """
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(inputs, filters, is_training, strides,
                    use_projection=True, data_format=data_format,
                    dropblock_keep_prob=dropblock_keep_prob,
                    dropblock_size=dropblock_size)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, 1,
                      data_format=data_format,
                      dropblock_keep_prob=dropblock_keep_prob,
                      dropblock_size=dropblock_size)

  return tf.identity(inputs, name)


def resnet_v1_generator(block_fn, layers, width_multiplier,
                        cifar_stem=False, data_format='channels_last',
                        dropblock_keep_probs=None, dropblock_size=None):
  """Generator for ResNet v1 models.

  Args:
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 4 `int`s denoting the number of blocks to include in each
      of the 4 block groups. Each group consists of blocks that take inputs of
      the same resolution.
    width_multiplier: `int` width multiplier for network.
    cifar_stem: `bool` If True, use a 3x3 conv without strides or pooling as
      stem.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    dropblock_keep_probs: `list` of 4 elements denoting keep_prob of DropBlock
      for each block group. None indicates no DropBlock for the corresponding
      block group.
    dropblock_size: `int`: size parameter of DropBlock.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.

  Raises:
    if dropblock_keep_probs is not 'None' or a list with len 4.
  """
  if dropblock_keep_probs is None:
    dropblock_keep_probs = [None] * 4
  if not isinstance(dropblock_keep_probs,
                    list) or len(dropblock_keep_probs) != 4:
    raise ValueError('dropblock_keep_probs is not valid:', dropblock_keep_probs)

  def model(inputs, is_training):
    """Creation of the model graph."""
    if cifar_stem:
      inputs = conv2d_fixed_padding(
          inputs=inputs, filters=64 * width_multiplier, kernel_size=3,
          strides=1, data_format=data_format)
      inputs = tf.identity(inputs, 'initial_conv')
      inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
      inputs = tf.identity(inputs, 'initial_max_pool')
    else:
      if FLAGS.sk_ratio > 0:  # Use ResNet-D (https://arxiv.org/abs/1812.01187)
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=64 * width_multiplier // 2, kernel_size=3,
            strides=2, data_format=data_format)
        inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=64 * width_multiplier // 2, kernel_size=3,
            strides=1, data_format=data_format)
        inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=64 * width_multiplier, kernel_size=3,
            strides=1, data_format=data_format)
      else:
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=64 * width_multiplier, kernel_size=7,
            strides=2, data_format=data_format)
      inputs = tf.identity(inputs, 'initial_conv')
      inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

      inputs = tf.layers.max_pooling2d(
          inputs=inputs, pool_size=3, strides=2, padding='SAME',
          data_format=data_format)
      inputs = tf.identity(inputs, 'initial_max_pool')

    def filter_trainable_variables(trainable_variables, after_block):
      """Add new trainable variables for the immediate precedent block."""
      if after_block == 0:
        trainable_variables[after_block] = tf.trainable_variables()
      else:
        trainable_variables[after_block] = []
        for var in tf.trainable_variables():
          to_keep = True
          for j in range(after_block):
            if var in trainable_variables[j]:
              to_keep = False
              break
          if to_keep:
            trainable_variables[after_block].append(var)

    def add_to_collection(trainable_variables, prefix):
      """Put variables into graph collection."""
      for after_block, variables in trainable_variables.items():
        collection = prefix + str(after_block)
        for var in variables:
          tf.add_to_collection(collection, var)

    trainable_variables = {}
    filter_trainable_variables(trainable_variables, after_block=0)
    if FLAGS.train_mode == 'finetune' and FLAGS.fine_tune_after_block == 0:
      inputs = tf.stop_gradient(inputs)

    inputs = block_group(
        inputs=inputs, filters=64 * width_multiplier, block_fn=block_fn,
        blocks=layers[0], strides=1, is_training=is_training,
        name='block_group1', data_format=data_format,
        dropblock_keep_prob=dropblock_keep_probs[0],
        dropblock_size=dropblock_size)

    filter_trainable_variables(trainable_variables, after_block=1)
    if FLAGS.train_mode == 'finetune' and FLAGS.fine_tune_after_block == 1:
      inputs = tf.stop_gradient(inputs)

    inputs = block_group(
        inputs=inputs, filters=128 * width_multiplier, block_fn=block_fn,
        blocks=layers[1], strides=2, is_training=is_training,
        name='block_group2', data_format=data_format,
        dropblock_keep_prob=dropblock_keep_probs[1],
        dropblock_size=dropblock_size)

    filter_trainable_variables(trainable_variables, after_block=2)
    if FLAGS.train_mode == 'finetune' and FLAGS.fine_tune_after_block == 2:
      inputs = tf.stop_gradient(inputs)

    inputs = block_group(
        inputs=inputs, filters=256 * width_multiplier, block_fn=block_fn,
        blocks=layers[2], strides=2, is_training=is_training,
        name='block_group3', data_format=data_format,
        dropblock_keep_prob=dropblock_keep_probs[2],
        dropblock_size=dropblock_size)

    filter_trainable_variables(trainable_variables, after_block=3)
    if FLAGS.train_mode == 'finetune' and FLAGS.fine_tune_after_block == 3:
      inputs = tf.stop_gradient(inputs)

    inputs = block_group(
        inputs=inputs, filters=512 * width_multiplier, block_fn=block_fn,
        blocks=layers[3], strides=2, is_training=is_training,
        name='block_group4', data_format=data_format,
        dropblock_keep_prob=dropblock_keep_probs[3],
        dropblock_size=dropblock_size)

    filter_trainable_variables(trainable_variables, after_block=4)
    if FLAGS.train_mode == 'finetune' and FLAGS.fine_tune_after_block == 4:
      inputs = tf.stop_gradient(inputs)

    if data_format == 'channels_last':
      inputs = tf.reduce_mean(inputs, [1, 2])
    else:
      inputs = tf.reduce_mean(inputs, [2, 3])
    inputs = tf.identity(inputs, 'final_avg_pool')

    # filter_trainable_variables(trainable_variables, after_block=5)
    add_to_collection(trainable_variables, 'trainable_variables_inblock_')

    return inputs

  return model


def resnet_v1(resnet_depth, width_multiplier,
              cifar_stem=False, data_format='channels_last',
              dropblock_keep_probs=None, dropblock_size=None):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
      34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)

  params = model_params[resnet_depth]
  return resnet_v1_generator(
      params['block'], params['layers'], width_multiplier,
      cifar_stem=cifar_stem,
      dropblock_keep_probs=dropblock_keep_probs,
      dropblock_size=dropblock_size,
      data_format=data_format)
