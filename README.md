# SimCLR - A Simple Framework for Contrastive Learning of Visual Representations

<span style="color: red"><strong>News! </strong></span> We have released a TF2 implementation of SimCLR (along with converted checkpoints in TF2), they are in <a href="tf2/">tf2/ folder</a>.

<span style="color: red"><strong>News! </strong></span> Colabs for <a href="https://arxiv.org/abs/2011.02803">Intriguing Properties of Contrastive Losses</a> are added, see <a href="colabs/intriguing_properties/">here</a>.

<div align="center">
  <img width="50%" alt="SimCLR Illustration" src="https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s1600/image4.gif">
</div>
<div align="center">
  An illustration of SimCLR (from <a href="https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html">our blog here</a>).
</div>

## Pre-trained models for SimCLRv2
<a href="colabs/finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

We opensourced total 65 pretrained models here, corresponding to those in Table 1 of the <a href="https://arxiv.org/abs/2006.10029">SimCLRv2</a> paper:

|   Depth | Width   | SK    |   Param (M) |   F-T (1%) |   F-T(10%) |   F-T(100%) |   Linear eval |   Supervised |
|--------:|--------:|------:|--------:|-------------:|--------------:|---------------:|-----------------:|--------------:|
|      50 | 1X      | False |    24 |         57.9 |          68.4 |           76.3 |             71.7 |          76.6 |
|      50 | 1X      | True  |    35 |         64.5 |          72.1 |           78.7 |             74.6 |          78.5 |
|      50 | 2X      | False |    94   |         66.3 |          73.9 |           79.1 |             75.6 |          77.8 |
|      50 | 2X      | True  |   140 |         70.6 |          77.0   |           81.3 |             77.7 |          79.3 |
|     101 | 1X      | False |    43 |         62.1 |          71.4 |           78.2 |             73.6 |          78.0   |
|     101 | 1X      | True  |    65 |         68.3 |          75.1 |           80.6 |             76.3 |          79.6 |
|     101 | 2X      | False |   170   |         69.1 |          75.8 |           80.7 |             77.0   |          78.9 |
|     101 | 2X      | True  |   257 |         73.2 |          78.8 |           82.4 |             79.0   |          80.1 |
|     152 | 1X      | False |    58 |         64.0   |          73.0   |           79.3 |             74.5 |          78.3 |
|     152 | 1X      | True  |    89 |         70.0   |          76.5 |           81.3 |             77.2 |          79.9 |
|     152 | 2X      | False |   233 |         70.2 |          76.6 |           81.1 |             77.4 |          79.1 |
|     152 | 2X      | True  |   354 |         74.2 |          79.4 |           82.9 |             79.4 |          80.4 |
|     152 | 3X      | True  |   795 |         74.9 |          80.1 |           83.1 |             79.8 |          80.5 |

These checkpoints are stored in Google Cloud Storage:

* Pretrained SimCLRv2 models (with linear eval head): [gs://simclr-checkpoints/simclrv2/pretrained](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/pretrained)
* Fine-tuned SimCLRv2 models on 1% of labels: [gs://simclr-checkpoints/simclrv2/finetuned_1pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/finetuned_1pct)
* Fine-tuned SimCLRv2 models on 10% of labels: [gs://simclr-checkpoints/simclrv2/finetuned_10pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/finetuned_10pct)
* Fine-tuned SimCLRv2 models on 100% of labels: [gs://simclr-checkpoints/simclrv2/finetuned_100pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/finetuned_100pct)
* Supervised models with the same architectures: [gs://simclr-checkpoints/simclrv2/supervised](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/supervised)
* The distilled / self-trained models (after fine-tuning) are also provided:
  * [gs://simclr-checkpoints/simclrv2/distill_1pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/distill_1pct)
  * [gs://simclr-checkpoints/simclrv2/distill_10pct](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/distill_10pct)

We also provide examples on how to use the checkpoints in `colabs/` folder.

## Pre-trained models for SimCLRv1

The pre-trained models (base network with linear classifier layer) can be found below. Note that for these SimCLRv1 checkpoints, the projection head is not available.

|                             Model checkpoint and hub-module                             |     ImageNet Top-1     |
|-----------------------------------------------------------------------------------------|------------------------|
|[ResNet50 (1x)](https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_1x.zip) |          69.1          |
|[ResNet50 (2x)](https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_2x.zip) |          74.2          |
|[ResNet50 (4x)](https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_4x.zip) |          76.6          |

Additional SimCLRv1 checkpoints are available: [gs://simclr-checkpoints/simclrv1](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv1).

A note on the signatures of the TensorFlow Hub module: `default` is the representation output of the base network; `logits_sup` is the supervised classification logits for ImageNet 1000 categories. Others (e.g. `initial_max_pool`, `block_group1`) are middle layers of ResNet; refer to resnet.py for the specifics. See this [tutorial](https://www.tensorflow.org/hub/tf1_hub_module) for additional information regarding use of TensorFlow Hub modules.

## Enviroment setup

Our models are trained with TPUs. It is recommended to run distributed training with TPUs when using our code for pretraining.

Our code can also run on a *single* GPU. It does not support multi-GPUs, for reasons such as global BatchNorm and contrastive loss across cores.

The code is compatible with both TensorFlow v1 and v2. See requirements.txt for all prerequisites, and you can also install them using the following command.

```
pip install -r requirements.txt
```

## Pretraining

To pretrain the model on CIFAR-10 with a *single* GPU, try the following command:

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

To fine-tune a linear head (with a single GPU), try the following command:

```
python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 \
  --train_epochs=100 --train_batch_size=512 --warmup_epochs=0 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --checkpoint=/tmp/simclr_test --model_dir=/tmp/simclr_test_ft --use_tpu=False
```

You can check the results using tensorboard, such as

```
python -m tensorboard.main --logdir=/tmp/simclr_test
```

As a reference, the above runs on CIFAR-10 should give you around 91% accuracy, though it can be further optimized.

For fine-tuning a linear head on ImageNet using Cloud TPUs, first set the `CHKPT_DIR` to pretrained model dir and set a new `MODEL_DIR`, then use the following command:

```
python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=1e-6 \
  --train_epochs=90 --train_batch_size=4096 --warmup_epochs=0 \
  --dataset=imagenet2012 --image_size=224 --eval_split=validation \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR \
  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0
```

As a reference, the above runs on ImageNet should give you around 64.5% accuracy.

## Semi-supervised learning and fine-tuning the whole network

You can access 1% and 10% ImageNet subsets used for semi-supervised learning via [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012_subset): simply set `dataset=imagenet2012_subset/1pct` and `dataset=imagenet2012_subset/10pct` in the command line for fine-tuning on these subsets.

You can also find image IDs of these subsets in `imagenet_subsets/`.

To fine-tune the whole network on ImageNet (1% of labels), refer to the following command:

```
python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=-1 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head_supervised)' \
  --global_bn=True --optimizer=lars --learning_rate=0.005 \
  --learning_rate_scaling=sqrt --weight_decay=0 \
  --train_epochs=60 --train_batch_size=1024 --warmup_epochs=0 \
  --dataset=imagenet2012_subset/1pct --image_size=224 --eval_split=validation \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR \
  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0 \
  --num_proj_layers=3 --ft_proj_selector=1
```

Set the `checkpoint` to those that are only pre-trained but not fine-tuned. Given that SimCLRv1 checkpoints do not contain projection head, it is recommended to run with SimCLRv2 checkpoints (you can still run with SimCLRv1 checkpoints, but `variable_schema` needs to exclude `head`). The `num_proj_layers` and `ft_proj_selector` need to be adjusted accordingly following SimCLRv2 paper to obtain best performances.

## Other resources

### Model conversion to Pytorch format

This [repo](https://github.com/tonylins/simclr-converter) provides a solution for converting the pretrained SimCLRv1 Tensorflow checkpoints into Pytorch ones.

This [repo](https://github.com/Separius/SimCLRv2-Pytorch) provides a solution for converting the pretrained SimCLRv2 Tensorflow checkpoints into Pytorch ones.

### Other *non-offical* / *unverified* implementations

(Feel free to share your implementation by creating an issue)

Implementations in PyTorch:
* [leftthomas](https://github.com/leftthomas/SimCLR)
* [AndrewAtanov](https://github.com/AndrewAtanov/simclr-pytorch)
* [ae-foster](https://github.com/ae-foster/pytorch-simclr)
* [Spijkervet](https://github.com/Spijkervet/SimCLR)
* [williamFalcon](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#simclr)

Implementations in Tensorflow 2 / Keras (official TF2 implementation was added in <a href="tf2/">tf2/ folder</a>):
* [sayakpaul](https://github.com/sayakpaul/SimCLR-in-TensorFlow-2)
* [mwdhont](https://github.com/mwdhont/SimCLRv1-keras-tensorflow)

## Known issues

* **Batch size**: original results of SimCLR were tuned under a large batch size (i.e. 4096), which leads to suboptimal results when training using a smaller batch size. However, with a good set of hyper-parameters (mainly learning rate, temperature, projection head depth), small batch sizes can yield results that are on par with large batch sizes (e.g., see Table 2 in [this paper](https://arxiv.org/pdf/2011.02803.pdf)).

* **Pretrained models / Checkpoints**: SimCLRv1 and SimCLRv2 are pretrained with different weight decays, so the pretrained models from the two versions have very different weight norm scales (convolutional weights in SimCLRv1 ResNet-50 are on average 16.8X of that in SimCLRv2). For fine-tuning the pretrained models from both versions, it is fine if you use an LARS optimizer, but it requires very different hyperparameters (e.g. learning rate, weight decay) if you use the momentum optimizer. So for the latter case, you may want to either search for very different hparams according to which version used, or re-scale th weight (i.e. conv `kernel` parameters of `base_model` in the checkpoints) to make sure they're roughly in the same scale.

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
