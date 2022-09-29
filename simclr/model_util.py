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
"""Network architectures related functions used in SimCLR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl import flags

import resnet
from lars_optimizer import LARSOptimizer

import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


def add_weight_decay(adjust_per_optimizer=True):
  """Compute weight decay from flags."""
  if adjust_per_optimizer and 'lars' in FLAGS.optimizer:
    # Weight decay are taking care of by optimizer for these cases.
    # Except for supervised head, which will be added here.
    l2_losses = [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                 if 'head_supervised' in v.name and 'bias' not in v.name]
    if l2_losses:
      tf.losses.add_loss(
          FLAGS.weight_decay * tf.add_n(l2_losses),
          tf.GraphKeys.REGULARIZATION_LOSSES)
    return

  l2_losses = [tf.nn.l2_loss(v) for v in tf.trainable_variables()
               if 'batch_normalization' not in v.name]
  tf.losses.add_loss(
      FLAGS.weight_decay * tf.add_n(l2_losses),
      tf.GraphKeys.REGULARIZATION_LOSSES)


def get_train_steps(num_examples):
  """Determine the number of training steps."""
  return FLAGS.train_steps or (
      num_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1)


def learning_rate_schedule(base_learning_rate, num_examples):
  """Build learning rate schedule."""
  global_step = tf.train.get_or_create_global_step()
  warmup_steps = int(round(
      FLAGS.warmup_epochs * num_examples // FLAGS.train_batch_size))
  if FLAGS.learning_rate_scaling == 'linear':
    scaled_lr = base_learning_rate * FLAGS.train_batch_size / 256.
  elif FLAGS.learning_rate_scaling == 'sqrt':
    scaled_lr = base_learning_rate * math.sqrt(FLAGS.train_batch_size)
  else:
    raise ValueError('Unknown learning rate scaling {}'.format(
        FLAGS.learning_rate_scaling))
  learning_rate = (tf.to_float(global_step) / int(warmup_steps) * scaled_lr
                   if warmup_steps else scaled_lr)

  # Cosine decay learning rate schedule
  total_steps = get_train_steps(num_examples)
  learning_rate = tf.where(
      global_step < warmup_steps, learning_rate,
      tf.train.cosine_decay(
          scaled_lr,
          global_step - warmup_steps,
          total_steps - warmup_steps))

  return learning_rate


def get_optimizer(learning_rate):
  """Returns an optimizer."""
  if FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, FLAGS.momentum, use_nesterov=True)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate)
  elif FLAGS.optimizer == 'lars':
    optimizer = LARSOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay,
        exclude_from_weight_decay=['batch_normalization', 'bias',
                                   'head_supervised'])
  else:
    raise ValueError('Unknown optimizer {}'.format(FLAGS.optimizer))

  if FLAGS.use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)
  return optimizer


def linear_layer(x,
                 is_training,
                 num_classes,
                 use_bias=True,
                 use_bn=False,
                 name='linear_layer'):
  """Linear head for linear evaluation.

  Args:
    x: hidden state tensor of shape (bsz, dim).
    is_training: boolean indicator for training or test.
    num_classes: number of classes.
    use_bias: whether or not to use bias.
    use_bn: whether or not to use BN for output units.
    name: the name for variable scope.

  Returns:
    logits of shape (bsz, num_classes)
  """
  assert x.shape.ndims == 2, x.shape
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    x = tf.layers.dense(
        inputs=x,
        units=num_classes,
        use_bias=use_bias and not use_bn,
        kernel_initializer=tf.random_normal_initializer(stddev=.01))
    if use_bn:
      x = resnet.batch_norm_relu(x, is_training, relu=False, center=use_bias)
    x = tf.identity(x, '%s_out' % name)
  return x


def projection_head(hiddens, is_training, name='head_contrastive'):
  """Head for projecting hiddens fo contrastive loss."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    mid_dim = hiddens.shape[-1]
    out_dim = FLAGS.proj_out_dim
    hiddens_list = [hiddens]
    if FLAGS.proj_head_mode == 'none':
      pass  # directly use the output hiddens as hiddens.
    elif FLAGS.proj_head_mode == 'linear':
      hiddens = linear_layer(
          hiddens, is_training, out_dim,
          use_bias=False, use_bn=True, name='l_0')
      hiddens_list.append(hiddens)
    elif FLAGS.proj_head_mode == 'nonlinear':
      for j in range(FLAGS.num_proj_layers):
        if j != FLAGS.num_proj_layers - 1:
          # for the middle layers, use bias and relu for the output.
          dim, bias_relu = mid_dim, True
        else:
          # for the final layer, neither bias nor relu is used.
          dim, bias_relu = FLAGS.proj_out_dim, False
        hiddens = linear_layer(
            hiddens, is_training, dim,
            use_bias=bias_relu, use_bn=True, name='nl_%d'%j)
        hiddens = tf.nn.relu(hiddens) if bias_relu else hiddens
        hiddens_list.append(hiddens)
    else:
      raise ValueError('Unknown head projection mode {}'.format(
          FLAGS.proj_head_mode))
    if FLAGS.train_mode == 'pretrain':
      # take the projection head output during pre-training.
      hiddens = hiddens_list[-1]
    else:
      # for checkpoint compatibility, whole projection head is built here.
      # but you can select part of projection head during fine-tuning.
      hiddens = hiddens_list[FLAGS.ft_proj_selector]
  return hiddens


def supervised_head(hiddens, num_classes, is_training, name='head_supervised'):
  """Add supervised head & also add its variables to inblock collection."""
  with tf.variable_scope(name):
    logits = linear_layer(hiddens, is_training, num_classes)
  for var in tf.trainable_variables():
    if var.name.startswith(name):
      tf.add_to_collection('trainable_variables_inblock_5', var)
  return logits
