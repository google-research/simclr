python3 run.py --dataset=bmw --data_dir="/mnt/data" --eval_split=test \
--use_tpu=False --model_dir=logs --mode=train_then_eval --train_mode=pretrain --image_size=224 \
--image_size=224 --train_batch_size=128 --lineareval_while_pretraining=False --train_test_ratio=0.3 \
--eval_batch_size=128 --load_existing_split=False --min_fraction_anomalies=0.15 --run_id= \
--gpus="GPU:1,GPU:2,GPU:4,GPU:5"


# --train_mode= pretrain or finetune
# it's (obviously) necessary to pretrain first because the models loads a checkpoint.

# I guess for our project we would only pretrain the model

# we then need to think about how to adapt the task for anomaly detection
# instead of a projection head, I guess we could use Conv2DTranspose to restore the original image
# and then compute a reconstruction loss over input and output


# steps
# import dataset to be class of TFDS builder
# pretrain with that model
# see what happens
# add upscaling head and compute reconstruction loss (finetune)