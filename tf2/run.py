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

import json
import math
import os

from absl import app
from absl import flags
from absl import logging
import data as data_lib
import metrics
import model as model_lib
import objective as obj_lib
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds



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

flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')

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
    'eval_steps', 0,
    'Number of steps to eval for. If not provided, evals over entire dataset.')

flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Batch size for eval.')

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

flags.DEFINE_bool('lineareval_while_pretraining', True,
                  'Whether to finetune supervised head while pretraining.')

flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for fine-tuning if a finetuning '
    'checkpoint does not already exist in model_dir.')

flags.DEFINE_bool(
    'zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linear head.')

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

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

flags.DEFINE_string(
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
    '0 means no projection head, and -1 means the final layer.')

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


def get_salient_tensors_dict(include_projection_head):
  """Returns a dictionary of tensors."""
  graph = tf.compat.v1.get_default_graph()
  result = {}
  for i in range(1, 5):
    result['block_group%d' % i] = graph.get_tensor_by_name(
        'resnet/block_group%d/block_group%d:0' % (i, i))
  result['initial_conv'] = graph.get_tensor_by_name(
      'resnet/initial_conv/Identity:0')
  result['initial_max_pool'] = graph.get_tensor_by_name(
      'resnet/initial_max_pool/Identity:0')
  result['final_avg_pool'] = graph.get_tensor_by_name('resnet/final_avg_pool:0')
  result['logits_sup'] = graph.get_tensor_by_name(
      'head_supervised/logits_sup:0')
  if include_projection_head:
    result['proj_head_input'] = graph.get_tensor_by_name(
        'projection_head/proj_head_input:0')
    result['proj_head_output'] = graph.get_tensor_by_name(
        'projection_head/proj_head_output:0')
  return result


def build_saved_model(model, include_projection_head=True):
  """Returns a tf.Module for saving to SavedModel."""

  class SimCLRModel(tf.Module):
    """Saved model for exporting to hub."""

    def __init__(self, model):
      self.model = model
      # This can't be called `trainable_variables` because `tf.Module` has
      # a getter with the same name.
      self.trainable_variables_list = model.trainable_variables

    @tf.function
    def __call__(self, inputs, trainable):
      self.model(inputs, training=trainable)
      return get_salient_tensors_dict(include_projection_head)

  module = SimCLRModel(model)
  input_spec = tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)
  module.__call__.get_concrete_function(input_spec, trainable=True)
  module.__call__.get_concrete_function(input_spec, trainable=False)
  return module


def save(model, global_step):
  """Export as SavedModel for finetuning and inference."""
  saved_model = build_saved_model(model)
  export_dir = os.path.join(FLAGS.model_dir, 'saved_model')
  checkpoint_export_dir = os.path.join(export_dir, str(global_step))
  if tf.io.gfile.exists(checkpoint_export_dir):
    tf.io.gfile.rmtree(checkpoint_export_dir)
  tf.saved_model.save(saved_model, checkpoint_export_dir)

  if FLAGS.keep_hub_module_max > 0:
    # Delete old exported SavedModels.
    exported_steps = []
    for subdir in tf.io.gfile.listdir(export_dir):
      if not subdir.isdigit():
        continue
      exported_steps.append(int(subdir))
    exported_steps.sort()
    for step_to_delete in exported_steps[:-FLAGS.keep_hub_module_max]:
      tf.io.gfile.rmtree(os.path.join(export_dir, str(step_to_delete)))


def try_restore_from_checkpoint(model, global_step, optimizer):
  """Restores the latest ckpt if it exists, otherwise check FLAGS.checkpoint."""
  checkpoint = tf.train.Checkpoint(
      model=model, global_step=global_step, optimizer=optimizer)
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=FLAGS.model_dir,
      max_to_keep=FLAGS.keep_checkpoint_max)
  latest_ckpt = checkpoint_manager.latest_checkpoint
  if latest_ckpt:
    # Restore model weights, global step, optimizer states
    logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
    checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
  elif FLAGS.checkpoint:
    # Restore model weights only, but not global step and optimizer states
    logging.info('Restoring from given checkpoint: %s', FLAGS.checkpoint)
    checkpoint_manager2 = tf.train.CheckpointManager(
        tf.train.Checkpoint(model=model),
        directory=FLAGS.model_dir,
        max_to_keep=FLAGS.keep_checkpoint_max)
    checkpoint_manager2.checkpoint.restore(FLAGS.checkpoint).expect_partial()
    if FLAGS.zero_init_logits_layer:
      model = checkpoint_manager2.checkpoint.model
      output_layer_parameters = model.supervised_head.trainable_weights
      logging.info('Initializing output layer parameters %s to zero',
                   [x.op.name for x in output_layer_parameters])
      for x in output_layer_parameters:
        x.assign(tf.zeros_like(x))

  return checkpoint_manager


def json_serializable(val):
  try:
    json.dumps(val)
    return True
  except TypeError:
    return False


def perform_evaluation(model, builder, eval_steps, ckpt, strategy, topology):
  """Perform evaluation."""
  if FLAGS.train_mode == 'pretrain' and not FLAGS.lineareval_while_pretraining:
    logging.info('Skipping eval during pretraining without linear eval.')
    return
  # Build input pipeline.
  ds = data_lib.build_distributed_dataset(builder, FLAGS.eval_batch_size, False,
                                          strategy, topology)
  summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

  # Build metrics.
  with strategy.scope():
    regularization_loss = tf.keras.metrics.Mean('eval/regularization_loss')
    label_top_1_accuracy = tf.keras.metrics.Accuracy(
        'eval/label_top_1_accuracy')
    label_top_5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
        5, 'eval/label_top_5_accuracy')
    all_metrics = [
        regularization_loss, label_top_1_accuracy, label_top_5_accuracy
    ]

    # Restore checkpoint.
    logging.info('Restoring from %s', ckpt)
    checkpoint = tf.train.Checkpoint(
        model=model, global_step=tf.Variable(0, dtype=tf.int64))
    checkpoint.restore(ckpt).expect_partial()
    global_step = checkpoint.global_step
    logging.info('Performing eval at step %d', global_step.numpy())

  def single_step(features, labels):
    _, supervised_head_outputs = model(features, training=False)
    assert supervised_head_outputs is not None
    outputs = supervised_head_outputs
    l = labels['labels']
    metrics.update_finetune_metrics_eval(label_top_1_accuracy,
                                         label_top_5_accuracy, outputs, l)
    reg_loss = model_lib.add_weight_decay(model, adjust_per_optimizer=True)
    regularization_loss.update_state(reg_loss)

  with strategy.scope():

    @tf.function
    def run_single_step(iterator):
      images, labels = next(iterator)
      features, labels = images, {'labels': labels}
      strategy.run(single_step, (features, labels))

    iterator = iter(ds)
    for i in range(eval_steps):
      run_single_step(iterator)
      logging.info('Completed eval for %d / %d steps', i + 1, eval_steps)
    logging.info('Finished eval for %s', ckpt)

  # Write summaries
  cur_step = global_step.numpy()
  logging.info('Writing summaries for %d step', cur_step)
  with summary_writer.as_default():
    metrics.log_and_write_metrics_to_summary(all_metrics, cur_step)
    summary_writer.flush()

  # Record results as JSON.
  result_json_path = os.path.join(FLAGS.model_dir, 'result.json')
  result = {metric.name: metric.result().numpy() for metric in all_metrics}
  result['global_step'] = global_step.numpy()
  logging.info(result)
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  result_json_path = os.path.join(
      FLAGS.model_dir, 'result_%d.json'%result['global_step'])
  with tf.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  flag_json_path = os.path.join(FLAGS.model_dir, 'flags.json')
  with tf.io.gfile.GFile(flag_json_path, 'w') as f:
    serializable_flags = {}
    for key, val in FLAGS.flag_values_dict().items():
      # Some flag value types e.g. datetime.timedelta are not json serializable,
      # filter those out.
      if json_serializable(val):
        serializable_flags[key] = val
    json.dump(serializable_flags, f)

  # Export as SavedModel for finetuning and inference.
  save(model, global_step=result['global_step'])

  return result


def _restore_latest_or_from_pretrain(checkpoint_manager):
  """Restores the latest ckpt if training already.

  Or restores from FLAGS.checkpoint if in finetune mode.

  Args:
    checkpoint_manager: tf.traiin.CheckpointManager.
  """
  latest_ckpt = checkpoint_manager.latest_checkpoint
  if latest_ckpt:
    # The model is not build yet so some variables may not be available in
    # the object graph. Those are lazily initialized. To suppress the warning
    # in that case we specify `expect_partial`.
    logging.info('Restoring from %s', latest_ckpt)
    checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
  elif FLAGS.train_mode == 'finetune':
    # Restore from pretrain checkpoint.
    assert FLAGS.checkpoint, 'Missing pretrain checkpoint.'
    logging.info('Restoring from %s', FLAGS.checkpoint)
    checkpoint_manager.checkpoint.restore(FLAGS.checkpoint).expect_partial()
    # TODO(iamtingchen): Can we instead use a zeros initializer for the
    # supervised head?
    if FLAGS.zero_init_logits_layer:
      model = checkpoint_manager.checkpoint.model
      output_layer_parameters = model.supervised_head.trainable_weights
      logging.info('Initializing output layer parameters %s to zero',
                   [x.op.name for x in output_layer_parameters])
      for x in output_layer_parameters:
        x.assign(tf.zeros_like(x))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


  builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)
  builder.download_and_prepare()
  num_train_examples = builder.info.splits[FLAGS.train_split].num_examples
  num_eval_examples = builder.info.splits[FLAGS.eval_split].num_examples
  num_classes = builder.info.features['label'].num_classes

  train_steps = model_lib.get_train_steps(num_train_examples)
  eval_steps = FLAGS.eval_steps or int(
      math.ceil(num_eval_examples / FLAGS.eval_batch_size))
  epoch_steps = int(round(num_train_examples / FLAGS.train_batch_size))

  logging.info('# train examples: %d', num_train_examples)
  logging.info('# train_steps: %d', train_steps)
  logging.info('# eval examples: %d', num_eval_examples)
  logging.info('# eval steps: %d', eval_steps)

  checkpoint_steps = (
      FLAGS.checkpoint_steps or (FLAGS.checkpoint_epochs * epoch_steps))

  topology = None
  if FLAGS.use_tpu:
    if FLAGS.tpu_name:
      cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    else:
      cluster = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.master)
    tf.config.experimental_connect_to_cluster(cluster)
    topology = tf.tpu.experimental.initialize_tpu_system(cluster)
    logging.info('Topology:')
    logging.info('num_tasks: %d', topology.num_tasks)
    logging.info('num_tpus_per_task: %d', topology.num_tpus_per_task)
    strategy = tf.distribute.TPUStrategy(cluster)

  else:
    # For (multiple) GPUs.
    strategy = tf.distribute.MirroredStrategy()
    logging.info('Running using MirroredStrategy on %d replicas',
                 strategy.num_replicas_in_sync)

  with strategy.scope():
    model = model_lib.Model(num_classes)

  if FLAGS.mode == 'eval':
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir, min_interval_secs=15):
      result = perform_evaluation(model, builder, eval_steps, ckpt, strategy,
                                  topology)
      if result['global_step'] >= train_steps:
        logging.info('Eval complete. Exiting...')
        return
  else:
    summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)
    with strategy.scope():
      # Build input pipeline.
      ds = data_lib.build_distributed_dataset(builder, FLAGS.train_batch_size,
                                              True, strategy, topology)

      # Build LR schedule and optimizer.
      learning_rate = model_lib.WarmUpAndCosineDecay(FLAGS.learning_rate,
                                                     num_train_examples)
      optimizer = model_lib.build_optimizer(learning_rate)

      # Build metrics.
      all_metrics = []  # For summaries.
      weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
      total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
      all_metrics.extend([weight_decay_metric, total_loss_metric])
      if FLAGS.train_mode == 'pretrain':
        contrast_loss_metric = tf.keras.metrics.Mean('train/contrast_loss')
        contrast_acc_metric = tf.keras.metrics.Mean('train/contrast_acc')
        contrast_entropy_metric = tf.keras.metrics.Mean(
            'train/contrast_entropy')
        all_metrics.extend([
            contrast_loss_metric, contrast_acc_metric, contrast_entropy_metric
        ])
      if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
        supervised_loss_metric = tf.keras.metrics.Mean('train/supervised_loss')
        supervised_acc_metric = tf.keras.metrics.Mean('train/supervised_acc')
        all_metrics.extend([supervised_loss_metric, supervised_acc_metric])

      # Restore checkpoint if available.
      checkpoint_manager = try_restore_from_checkpoint(
          model, optimizer.iterations, optimizer)

    steps_per_loop = checkpoint_steps

    def single_step(features, labels):
      with tf.GradientTape() as tape:
        # Log summaries on the last step of the training loop to match
        # logging frequency of other scalar summaries.
        #
        # Notes:
        # 1. Summary ops on TPUs get outside compiled so they do not affect
        #    performance.
        # 2. Summaries are recorded only on replica 0. So effectively this
        #    summary would be written once per host when should_record == True.
        # 3. optimizer.iterations is incremented in the call to apply_gradients.
        #    So we use  `iterations + 1` here so that the step number matches
        #    those of scalar summaries.
        # 4. We intentionally run the summary op before the actual model
        #    training so that it can run in parallel.
        should_record = tf.equal((optimizer.iterations + 1) % steps_per_loop, 0)
        with tf.summary.record_if(should_record):
          # Only log augmented images for the first tower.
          tf.summary.image(
              'image', features[:, :, :, :3], step=optimizer.iterations + 1)
        projection_head_outputs, supervised_head_outputs = model(
            features, training=True)
        loss = None
        if projection_head_outputs is not None:
          outputs = projection_head_outputs
          con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
              outputs,
              hidden_norm=FLAGS.hidden_norm,
              temperature=FLAGS.temperature,
              strategy=strategy)
          if loss is None:
            loss = con_loss
          else:
            loss += con_loss
          metrics.update_pretrain_metrics_train(contrast_loss_metric,
                                                contrast_acc_metric,
                                                contrast_entropy_metric,
                                                con_loss, logits_con,
                                                labels_con)
        if supervised_head_outputs is not None:
          outputs = supervised_head_outputs
          l = labels['labels']
          if FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            l = tf.concat([l, l], 0)
          sup_loss = obj_lib.add_supervised_loss(labels=l, logits=outputs)
          if loss is None:
            loss = sup_loss
          else:
            loss += sup_loss
          metrics.update_finetune_metrics_train(supervised_loss_metric,
                                                supervised_acc_metric, sup_loss,
                                                l, outputs)
        weight_decay = model_lib.add_weight_decay(
            model, adjust_per_optimizer=True)
        weight_decay_metric.update_state(weight_decay)
        loss += weight_decay
        total_loss_metric.update_state(loss)
        # The default behavior of `apply_gradients` is to sum gradients from all
        # replicas so we divide the loss by the number of replicas so that the
        # mean gradient is applied.
        loss = loss / strategy.num_replicas_in_sync
        logging.info('Trainable variables:')
        for var in model.trainable_variables:
          logging.info(var.name)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    with strategy.scope():

      @tf.function
      def train_multiple_steps(iterator):
        # `tf.range` is needed so that this runs in a `tf.while_loop` and is
        # not unrolled.
        for _ in tf.range(steps_per_loop):
          # Drop the "while" prefix created by tf.while_loop which otherwise
          # gets prefixed to every variable name. This does not affect training
          # but does affect the checkpoint conversion script.
          # TODO(b/161712658): Remove this.
          with tf.name_scope(''):
            images, labels = next(iterator)
            features, labels = images, {'labels': labels}
            strategy.run(single_step, (features, labels))

      global_step = optimizer.iterations
      cur_step = global_step.numpy()
      iterator = iter(ds)
      while cur_step < train_steps:
        # Calls to tf.summary.xyz lookup the summary writer resource which is
        # set by the summary writer's context manager.
        with summary_writer.as_default():
          train_multiple_steps(iterator)
          cur_step = global_step.numpy()
          checkpoint_manager.save(cur_step)
          logging.info('Completed: %d / %d steps', cur_step, train_steps)
          metrics.log_and_write_metrics_to_summary(all_metrics, cur_step)
          tf.summary.scalar(
              'learning_rate',
              learning_rate(tf.cast(global_step, dtype=tf.float32)),
              global_step)
          summary_writer.flush()
        for metric in all_metrics:
          metric.reset_states()
      logging.info('Training complete...')

    if FLAGS.mode == 'train_then_eval':
      perform_evaluation(model, builder, eval_steps,
                         checkpoint_manager.latest_checkpoint, strategy,
                         topology)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  # For outside compilation of summaries on TPU.
  tf.config.set_soft_device_placement(True)
  app.run(main)
