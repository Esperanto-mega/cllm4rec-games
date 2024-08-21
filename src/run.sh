#!/bin/sh

lambda_V=1
dataset=Beauty
save_dir=/datain/v-yinju/rqvae-zzx/models/cllm4rec/$dataset

accelerate launch --multi_gpu --num_processes=8 ~/cllm4rec-games/src/training.py --dataset $dataset --lambda_V $lambda_V

accelerate launch --multi_gpu --num_processes=8 ~/cllm4rec-games/src/finetuning.py \
  --dataset $dataset \
  --lambda_V $lambda_V \
  --batch_size 8 \
  --val_batch_size 256

python ~/cllm4rec-games/src/predict.py --dataset $dataset --lambda_V $lambda_V

cd ~/rq-llama
source instruments_train.sh
