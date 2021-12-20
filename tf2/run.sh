python3 run.py --dataset=pill --eval_split=train --use_tpu=False --model_dir=logs --train_mode=pretrain

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