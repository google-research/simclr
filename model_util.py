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

from absl import flags

import resnet

import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


def add_weight_decay(adjust_per_optimizer=True):
  """Compute weight decay from flags."""
  if adjust_per_optimizer and 'lars' in FLAGS.optimizer:
    # Weight decay are taking care of by optimizer for these cases.
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
  scaled_lr = base_learning_rate * FLAGS.train_batch_size / 256.
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
    if FLAGS.head_proj_mode == 'none':
      pass  # directly use the output hiddens as hiddens
    elif FLAGS.head_proj_mode == 'linear':
      hiddens = linear_layer(
          hiddens, is_training, FLAGS.head_proj_dim,
          use_bias=False, use_bn=True, name='l_0')
    elif FLAGS.head_proj_mode == 'nonlinear':
      hiddens = linear_layer(
          hiddens, is_training, hiddens.shape[-1],
          use_bias=True, use_bn=True, name='nl_0')
      for j in range(1, FLAGS.num_nlh_layers + 1):
        hiddens = tf.nn.relu(hiddens)
        hiddens = linear_layer(
            hiddens, is_training, FLAGS.head_proj_dim,
            use_bias=False, use_bn=True, name='nl_%d'%j)
    else:
      raise ValueError('Unknown head projection mode {}'.format(
          FLAGS.head_proj_mode))
  return hiddens


def supervised_head(hiddens, num_classes, is_training, name='head_supervised'):
  """Add supervised head & also add its variables to inblock collection."""
  with tf.variable_scope(name):
    logits = linear_layer(hiddens, is_training, num_classes)
  for var in tf.trainable_variables():
    if var.name.startswith(name):
      tf.add_to_collection('trainable_variables_inblock_5', var)
  return logits
