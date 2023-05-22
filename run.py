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
"""The main training pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
from absl import app
from absl import flags

import resnet
import data as data_lib
import model as model_lib
import model_util as model_util

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow_datasets as tfds
import tensorflow_hub as hub


FLAGS = flags.FLAGS


flags.DEFINE_float(
    'learning_rate', 0.3,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_enum(
    'learning_rate_scaling', 'linear', ['linear', 'sqrt'],
    'How to scale the learning rate as a function of batch size.')

flags.DEFINE_float(
    'warmup_epochs', 10,
    'Number of epochs of warmup.')

flags.DEFINE_float(
    'weight_decay', 1e-4,
    'Amount of weight decay to use.')

flags.DEFINE_float(
    'batch_norm_decay', 0.9,
    'Batch norm decay parameter.')

flags.DEFINE_integer(
    'train_batch_size', 512,
    'Batch size for training.')

flags.DEFINE_string(
    'train_split', 'train',
    'Split for training.')

flags.DEFINE_integer(
    'train_epochs', 100,
    'Number of epochs to train for.')

flags.DEFINE_integer(
    'train_steps', 0,
    'Number of steps to train for. If provided, overrides train_epochs.')

flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Batch size for eval.')

flags.DEFINE_integer(
    'train_summary_steps', 100,
    'Steps before saving training summaries. If 0, will not save.')

flags.DEFINE_integer(
    'checkpoint_epochs', 1,
    'Number of epochs between checkpoints/summaries.')

flags.DEFINE_integer(
    'checkpoint_steps', 0,
    'Number of steps between checkpoints/summaries. If provided, overrides '
    'checkpoint_epochs.')

flags.DEFINE_string(
    'eval_split', 'validation',
    'Split for evaluation.')

flags.DEFINE_string(
    'dataset', 'imagenet2012',
    'Name of a dataset.')

flags.DEFINE_bool(
    'cache_dataset', False,
    'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can '
    'improve performance.')

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for continued training or fine-tuning.')

flags.DEFINE_string(
    'variable_schema', '?!global_step',
    'This defines whether some variable from the checkpoint should be loaded.')

flags.DEFINE_bool(
    'zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linera head.')

flags.DEFINE_string(
    'master', None,
    'Address/name of the TensorFlow master to use. By default, use an '
    'in-process master.')

flags.DEFINE_string(
    'model_dir', None,
    'Model directory for training.')

flags.DEFINE_string(
    'data_dir', None,
    'Directory where dataset is stored.')

flags.DEFINE_bool(
    'use_tpu', True,
    'Whether to run on TPU.')

tf.flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

tf.flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

tf.flags.DEFINE_string(
    'gcp_project', None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

flags.DEFINE_enum(
    'optimizer', 'lars', ['momentum', 'adam', 'lars'],
    'Optimizer to use.')

flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum parameter.')

flags.DEFINE_string(
    'eval_name', None,
    'Name for eval.')

flags.DEFINE_integer(
    'keep_checkpoint_max', 5,
    'Maximum number of checkpoints to keep.')

flags.DEFINE_integer(
    'keep_hub_module_max', 1,
    'Maximum number of Hub modules to keep.')

flags.DEFINE_float(
    'temperature', 0.1,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_boolean(
    'hidden_norm', True,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_enum(
    'proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer(
    'proj_out_dim', 128,
    'Number of head projection dimension.')

flags.DEFINE_integer(
    'num_proj_layers', 3,
    'Number of non-linear head layers.')

flags.DEFINE_integer(
    'ft_proj_selector', 0,
    'Which layer of the projection head to use during fine-tuning. '
    '0 means throwing away the projection head, and -1 means the final layer.')

flags.DEFINE_boolean(
    'global_bn', True,
    'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_integer(
    'width_multiplier', 1,
    'Multiplier to change width of network.')

flags.DEFINE_integer(
    'resnet_depth', 50,
    'Depth of ResNet.')

flags.DEFINE_float(
    'sk_ratio', 0.,
    'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')

flags.DEFINE_float(
    'se_ratio', 0.,
    'If it is bigger than 0, it will enable SE.')

flags.DEFINE_integer(
    'image_size', 224,
    'Input image size.')

flags.DEFINE_float(
    'color_jitter_strength', 1.0,
    'The strength of color jittering.')

flags.DEFINE_boolean(
    'use_blur', True,
    'Whether or not to use Gaussian blur for augmentation during pretraining.')


def build_hub_module(model, num_classes, global_step, checkpoint_path):
  """Create TF-Hub module."""

  tags_and_args = [
      # The default graph is built with batch_norm, dropout etc. in inference
      # mode. This graph version is good for inference, not training.
      ([], {'is_training': False}),
      # A separate "train" graph builds batch_norm, dropout etc. in training
      # mode.
      (['train'], {'is_training': True}),
  ]

  def module_fn(is_training):
    """Function that builds TF-Hub module."""
    endpoints = {}
    inputs = tf.placeholder(
        tf.float32, [None, None, None, 3])
    with tf.variable_scope('base_model', reuse=tf.AUTO_REUSE):
      hiddens = model(inputs, is_training)
      for v in ['initial_conv', 'initial_max_pool', 'block_group1',
                'block_group2', 'block_group3', 'block_group4',
                'final_avg_pool']:
        endpoints[v] = tf.get_default_graph().get_tensor_by_name(
            'base_model/{}:0'.format(v))
    if FLAGS.train_mode == 'pretrain':
      hiddens_proj = model_util.projection_head(hiddens, is_training)
      endpoints['proj_head_input'] = hiddens
      endpoints['proj_head_output'] = hiddens_proj
    else:
      logits_sup = model_util.supervised_head(
          hiddens, num_classes, is_training)
      endpoints['logits_sup'] = logits_sup
    hub.add_signature(inputs=dict(images=inputs),
                      outputs=dict(endpoints, default=hiddens))

  # Drop the non-supported non-standard graph collection.
  drop_collections = ['trainable_variables_inblock_%d'%d for d in range(6)]
  spec = hub.create_module_spec(module_fn, tags_and_args, drop_collections)
  hub_export_dir = os.path.join(FLAGS.model_dir, 'hub')
  checkpoint_export_dir = os.path.join(hub_export_dir, str(global_step))
  if tf.io.gfile.exists(checkpoint_export_dir):
    # Do not save if checkpoint already saved.
    tf.io.gfile.rmtree(checkpoint_export_dir)
  spec.export(
      checkpoint_export_dir,
      checkpoint_path=checkpoint_path,
      name_transform_fn=None)

  if FLAGS.keep_hub_module_max > 0:
    # Delete old exported Hub modules.
    exported_steps = []
    for subdir in tf.io.gfile.listdir(hub_export_dir):
      if not subdir.isdigit():
        continue
      exported_steps.append(int(subdir))
    exported_steps.sort()
    for step_to_delete in exported_steps[:-FLAGS.keep_hub_module_max]:
      tf.io.gfile.rmtree(os.path.join(hub_export_dir, str(step_to_delete)))


def perform_evaluation(estimator, input_fn, eval_steps, model, num_classes,
                       checkpoint_path=None):
  """Perform evaluation.

  Args:
    estimator: TPUEstimator instance.
    input_fn: Input function for estimator.
    eval_steps: Number of steps for evaluation.
    model: Instance of transfer_learning.models.Model.
    num_classes: Number of classes to build model for.
    checkpoint_path: Path of checkpoint to evaluate.

  Returns:
    result: A Dict of metrics and their values.
  """
  if not checkpoint_path:
    checkpoint_path = estimator.latest_checkpoint()
  result = estimator.evaluate(
      input_fn, eval_steps, checkpoint_path=checkpoint_path,
      name=FLAGS.eval_name)

  # Record results as JSON.
  result_json_path = os.path.join(FLAGS.model_dir, 'result.json')
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  result_json_path = os.path.join(
      FLAGS.model_dir, 'result_%d.json'%result['global_step'])
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  flag_json_path = os.path.join(FLAGS.model_dir, 'flags.json')

  def json_serializable(val):
    try:
      json.dumps(val)
      return True
    except TypeError:
      return False

  with tf.io.gfile.GFile(flag_json_path, 'w') as f:
    serializable_flags = {}
    for key, val in FLAGS.flag_values_dict().items():
      # Some flag value types e.g. datetime.timedelta are not json serializable,
      # filter those out.
      if json_serializable(val):
        serializable_flags[key] = val
    json.dump(serializable_flags, f)

  # Save Hub module.
  build_hub_module(model, num_classes,
                   global_step=result['global_step'],
                   checkpoint_path=checkpoint_path)

  return result


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Enable training summary.
  if FLAGS.train_summary_steps > 0:
    tf.config.set_soft_device_placement(True)


  builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)
  builder.download_and_prepare()
  num_train_examples = builder.info.splits[FLAGS.train_split].num_examples
  num_eval_examples = builder.info.splits[FLAGS.eval_split].num_examples
  num_classes = builder.info.features['label'].num_classes

  train_steps = model_util.get_train_steps(num_train_examples)
  eval_steps = int(math.ceil(num_eval_examples / FLAGS.eval_batch_size))
  epoch_steps = int(round(num_train_examples / FLAGS.train_batch_size))

  resnet.BATCH_NORM_DECAY = FLAGS.batch_norm_decay
  model = resnet.resnet_v1(
      resnet_depth=FLAGS.resnet_depth,
      width_multiplier=FLAGS.width_multiplier,
      cifar_stem=FLAGS.image_size <= 32)

  checkpoint_steps = (
      FLAGS.checkpoint_steps or (FLAGS.checkpoint_epochs * epoch_steps))

  cluster = None
  if FLAGS.use_tpu and FLAGS.master is None:
    if FLAGS.tpu_name:
      cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    else:
      cluster = tf.distribute.cluster_resolver.TPUClusterResolver()
      tf.config.experimental_connect_to_cluster(cluster)
      tf.tpu.experimental.initialize_tpu_system(cluster)

  default_eval_mode = tf_estimator.tpu.InputPipelineConfig.PER_HOST_V1
  sliced_eval_mode = tf_estimator.tpu.InputPipelineConfig.SLICED
  run_config = tf_estimator.tpu.RunConfig(
      tpu_config=tf_estimator.tpu.TPUConfig(
          iterations_per_loop=checkpoint_steps,
          eval_training_input_configuration=sliced_eval_mode
          if FLAGS.use_tpu else default_eval_mode),
      model_dir=FLAGS.model_dir,
      save_summary_steps=checkpoint_steps,
      save_checkpoints_steps=checkpoint_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      master=FLAGS.master,
      cluster=cluster)
  estimator = tf_estimator.tpu.TPUEstimator(
      model_lib.build_model_fn(model, num_classes, num_train_examples),
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      use_tpu=FLAGS.use_tpu)

  if FLAGS.mode == 'eval':
    for ckpt in tf.train.checkpoints_iterator(
        run_config.model_dir, min_interval_secs=15):
      try:
        result = perform_evaluation(
            estimator=estimator,
            input_fn=data_lib.build_input_fn(builder, False),
            eval_steps=eval_steps,
            model=model,
            num_classes=num_classes,
            checkpoint_path=ckpt)
      except tf.errors.NotFoundError:
        continue
      if result['global_step'] >= train_steps:
        return
  else:
    estimator.train(
        data_lib.build_input_fn(builder, True), max_steps=train_steps)
    if FLAGS.mode == 'train_then_eval':
      perform_evaluation(
          estimator=estimator,
          input_fn=data_lib.build_input_fn(builder, False),
          eval_steps=eval_steps,
          model=model,
          num_classes=num_classes)


if __name__ == '__main__':
  tf.disable_v2_behavior()  # Disable eager mode when running with TF2.
  app.run(main)
