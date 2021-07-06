# TF2 implementation of SimCLR

This implementation is based on TensorFlow 2.x. We use `tf.keras` layers for building the model and use `tf.data` for our input pipeline. The model is trained using a [custom training loop](https://www.tensorflow.org/tutorials/distribute/custom_training) with `tf.distribute` on multiple TPUs.

<div align="center">
  <img width="50%" alt="SimCLR Illustration" src="https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s1600/image4.gif">
</div>
<div align="center">
  An illustration of SimCLR (from <a href="https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html">our blog here</a>).
</div>

<br/><br/>

## Pre-trained models for SimCLRv2
<a href="tf2/colabs/finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

We have converted the checkpoints for the TF1 models of SimCLR v1 and v2 to TF2 [SavedModel](https://www.tensorflow.org/guide/saved_model):

* Pretrained SimCLRv2 models (with linear eval head): [gs://simclr-checkpoints-tf2/simclrv2/pretrained](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv2/pretrained)
* Fine-tuned SimCLRv2 models on 1% of labels: [gs://simclr-checkpoints-tf2/simclrv2/finetuned_1pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv2/finetuned_1pct)
* Fine-tuned SimCLRv2 models on 10% of labels: [gs://simclr-checkpoints-tf2/simclrv2/finetuned_10pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv2/finetuned_10pct)
* Fine-tuned SimCLRv2 models on 100% of labels: [gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv2/finetuned_100pct)
* Supervised models with the same architectures: [gs://simclr-checkpoints-tf2/simclrv2/supervised](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv2/supervised)
* The distilled / self-trained models (after fine-tuning) are also provided:
  * [gs://simclr-checkpoints-tf2/simclrv2/distill_1pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv2/distill_1pct)
  * [gs://simclr-checkpoints-tf2/simclrv2/distill_10pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv2/distill_10pct)

We also provide examples on how to use the SavedModels in `colabs/` folder. In addition to the TF1 colabs we provide a `imagenet_results.ipynb` colab to verify results from SimCLR v1 and v2 papers for ImageNet.

## Pre-trained models for SimCLRv1

The pre-trained models (base network with linear classifier layer) can be found below. Note that for these SimCLRv1 checkpoints, the projection head is not available.

|                             SavedModel                                                                       |     ImageNet Top-1     |
|--------------------------------------------------------------------------------------------------------------|------------------------|
|[ResNet50 (1x)](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv1/pretrain/1x) |          69.1          |
|[ResNet50 (2x)](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv1/pretrain/2x) |          74.2          |
|[ResNet50 (4x)](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv1/pretrain/4x) |          76.6          |

Additional SimCLRv1 checkpoints are available: [gs://simclr-checkpoints-tf2/simclrv1](https://console.cloud.google.com/storage/browser/simclr-checkpoints-tf2/simclrv1).

A note on the signature of the TensorFlow SavedModel: `logits_sup` is the supervised classification logits for ImageNet 1000 categories. Others (e.g. `initial_max_pool`, `block_group1`) are middle layers of ResNet; refer to resnet.py for the specifics.

## Enviroment setup

Our models are trained with TPUs. It is recommended to run distributed training with TPUs when using our code for pretraining.

The code can be run on multiple GPUs by replacing `tf.distribute.TPUStrategy` with `tf.distribute.MirroredStrategy`. See the TensorFlow distributed training [guide](https://www.tensorflow.org/guide/distributed_training) for an overview of `tf.distribute`.

The code is compatible with TensorFlow 2.x. See requirements.txt for all prerequisites, and you can also install them using the following command.

```
pip install -r requirements.txt
```

## Pretraining

To pretrain the model on CIFAR-10 with CPU / 1 or more GPUs, try the following command:

```
python run.py --train_mode=pretrain \
  --train_batch_size=512 --train_epochs=1000 \
  --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --use_blur=False --color_jitter_strength=0.5 \
  --model_dir=/tmp/simclr_test --use_tpu=False
```

To pretrain the model on ImageNet with Cloud TPUs, first check out the [Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist) for basic information on how to use Google Cloud TPUs.

Once you have created virtual machine with Cloud TPUs, and pre-downloaded the ImageNet data for [tensorflow_datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012), please set the following enviroment variables:

```
TPU_NAME=<tpu-name>
STORAGE_BUCKET=gs://<storage-bucket>
DATA_DIR=$STORAGE_BUCKET/<path-to-tensorflow-dataset>
MODEL_DIR=$STORAGE_BUCKET/<path-to-store-checkpoints>
```

The following command can be used to pretrain a ResNet-50 on ImageNet (which reflects the default hyperparameters in our paper):

```
python run.py --train_mode=pretrain \
  --train_batch_size=4096 --train_epochs=100 --temperature=0.1 \
  --learning_rate=0.075 --learning_rate_scaling=sqrt --weight_decay=1e-4 \
  --dataset=imagenet2012 --image_size=224 --eval_split=validation \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0
```

A batch size of 4096 requires at least 32 TPUs. 100 epochs takes around 6 hours with 32 TPU v3s. Note that learning rate of 0.3 with `learning_rate_scaling=linear` is equivalent to that of 0.075 with `learning_rate_scaling=sqrt` when the batch size is 4096. However, using sqrt scaling allows it to train better when smaller batch size is used.

## Finetuning the linear head (linear eval)

You could simply set `--lineareval_while_pretraining=True` during pretraining, which will train the linear classifier as you pretrain the model. The `stop_gradient` operator is uesd to prevent backpropagating the label information to representations.

More conventionally, you can also finetune the linear head on top of a pretrained model after pretraining, as follows:

```
class Model(tf.keras.Model):
  def __init__(self, path):
    super(Model, self).__init__()
    # Load a pretrained SimCLR model.
    self.saved_model = tf.saved_model.load(path)
    # Linear head.
    self.dense_layer = tf.keras.layers.Dense(units=num_classes,
        name="head_supervised_new")
    self.optimizer = <your favorite optimizer>

  def call(self, x):
    with tf.GradientTape() as tape:
      # Use `trainable=False` since we do not wish to update batch norm
      # statistics of the loaded model. If finetuning everything, set this to
      # True.
      outputs = self.saved_model(x['image'], trainable=False)
      logits_t = self.dense_layer(outputs['final_avg_pool'])
      loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels = tf.one_hot(x['label'], num_classes), logits=logits_t))
      dense_layer_weights = self.dense_layer.trainable_weights
      print('Variables to train:', dense_layer_weights)
      # Note: We only compute gradients wrt the linear head. To finetune all
      # weights use self.trainable_weights instead.
      grads = tape.gradient(loss_t, dense_layer_weights)
      self.optimizer.apply_gradients(zip(grads, dense_layer_weights))
    return loss_t, x["image"], logits_t, x["label"]

model = Model("gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r50_1x_sk0/saved_model/")

# Use tf.function to speed up training. Remove this when debugging intermediate
# model activations.
@tf.function
def train_step(x):
  return model(x)
  
ds = build_dataset(...)
iterator = iter(ds)
for _ in range(num_steps):
  train_step(next(iterator))
```

Check the colab in `colabs/finetuning.ipynb` for a complete example.

## Semi-supervised learning and fine-tuning the whole network

You can access 1% and 10% ImageNet subsets used for semi-supervised learning via [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012_subset): simply set `dataset=imagenet2012_subset/1pct` and `dataset=imagenet2012_subset/10pct` in the command line for fine-tuning on these subsets.

You can also find image IDs of these subsets in `imagenet_subsets/`.

## Cite

[SimCLR paper](https://arxiv.org/abs/2002.05709):

```
@article{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020}
}
```

[SimCLRv2 paper](https://arxiv.org/abs/2006.10029):

```
@article{chen2020big,
  title={Big Self-Supervised Models are Strong Semi-Supervised Learners},
  author={Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2006.10029},
  year={2020}
}
```

## Disclaimer
This is not an official Google product.
