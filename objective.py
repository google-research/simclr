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
"""Contrastive loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from tensorflow.compiler.tf2xla.python import xla  # pylint: disable=g-direct-tensorflow-import

LARGE_NUM = 1e9


def add_supervised_loss(labels, logits, weights, **kwargs):
  """Compute loss for model and add it to loss collection."""
  return tf.losses.softmax_cross_entropy(labels, logits, weights, **kwargs)


def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=1.0,
                         tpu_context=None,
                         weights=1.0):
  """Compute loss for model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    tpu_context: context information for tpu.
    weights: a weighting number or vector.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
  """
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf.split(hidden, 2, 0)
  batch_size = tf.shape(hidden1)[0]

  # Gather hidden1/hidden2 across replicas and create local labels.
  if tpu_context is not None:
    hidden1_large = tpu_cross_replica_concat(hidden1, tpu_context)
    hidden2_large = tpu_cross_replica_concat(hidden2, tpu_context)
    enlarged_batch_size = tf.shape(hidden1_large)[0]
    # TODO(iamtingchen): more elegant way to convert u32 to s32 for replica_id.
    replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
    labels_idx = tf.range(batch_size) + replica_id * batch_size
    labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
    masks = tf.one_hot(labels_idx, enlarged_batch_size)
  else:
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

  logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
  logits_aa = logits_aa - masks * LARGE_NUM
  logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
  logits_bb = logits_bb - masks * LARGE_NUM
  logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
  logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

  loss_a = tf.losses.softmax_cross_entropy(
      labels, tf.concat([logits_ab, logits_aa], 1), weights=weights)
  loss_b = tf.losses.softmax_cross_entropy(
      labels, tf.concat([logits_ba, logits_bb], 1), weights=weights)
  loss = loss_a + loss_b

  return loss, logits_ab, labels


def tpu_cross_replica_concat(tensor, tpu_context=None):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    tpu_context: A `TPUContext`. If not set, CPU execution is assumed.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  if tpu_context is None or tpu_context.num_replicas <= 1:
    return tensor

  num_replicas = tpu_context.num_replicas

  with tf.name_scope('tpu_cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[xla.replica_id()]],
        updates=[tensor],
        shape=[num_replicas] + tensor.shape.as_list())

    # As every value is only present on one replica and 0 in all others, adding
    # them all together will result in the full tensor on all replicas.
    ext_tensor = tf.tpu.cross_replica_sum(ext_tensor)

    # Flatten the replica dimension.
    # The first dimension size will be: tensor.shape[0] * num_replicas
    # Using [-1] trick to support also scalar input.
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])
