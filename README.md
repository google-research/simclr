# SimCLR - A Simple Framework for Contrastive Learning of Visual Representations

## Pre-trained models

The pre-trained models (base network with linear classifier layer) can be found below.

|                             Model checkpoint and hub-module                             |     ImageNet Top-1     |
|-----------------------------------------------------------------------------------------|------------------------|
|[ResNet50 (1x)](https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_1x.zip) |          69.1          |
|[ResNet50 (2x)](https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_2x.zip) |          74.2          |
|[ResNet50 (4x)](https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_4x.zip) |          76.6          |


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
  --learning_rate=1.0 --weight_decay=1e-6 --temperature=0.5 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --use_blur=False --color_jitter_strength=0.5 \
  --model_dir=/tmp/simclr_test --use_tpu=False
```

To pretrain the model on ImageNet with Cloud TPUs, you should also set the following flags.

```
  --use_tpu=True
  --tpu_name=$TPU_NAME
```

Please see the [Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist) for how to use Cloud TPUs. More instruction on how to run with Cloud TPUs will be released soon!

## Finetuning

To fine-tune a linear head (with a single GPU), try the following command:

```
python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)LARSOptimizer|head)' \
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

## Others

### Semi-supervised learning

Image IDs of ImageNet 1% and 10% subsets used for semi-supervised learning can be found in `imagenet_subsets/`.

## Cite

Our [arXiv paper](https://arxiv.org/abs/2002.05709).

```
@article{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020}
}
```

## Disclaimer
This is not an official Google product.
