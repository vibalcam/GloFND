#!/bin/bash
####
# Train model locally
####

experiment=tuning

cuda_device=$1
i=$2
dataset="imagenet100"
loss_type=$3
glofnd=$4
alpha=$5
start_update=$6
lda_start=$6
checkpoint_id=$((lda_start - 1))
checkpoint_id=$(printf "%04d" $checkpoint_id)
lr_lda=$7
u_warmup=$8
init_quantile=False

timestamp=$(date +"%Y%m%d")
saved_path="${experiment}_${loss_type}_${glofnd}_${alpha}_${start_update}_${lda_start}_${lr_lda}_${u_warmup}"

first_char=${cuda_device:0:1}
port=$((10001 + first_char + lda_start))

# for i in {1..3}; do

    seed=$((12345 + i))
    batch_size=128

    # WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$cuda_device python -m debugpy --listen 5678 --wait-for-client train.py \
    CUDA_VISIBLE_DEVICES=$cuda_device python train.py \
        --lr=.075 --epochs=200 --batch-size=$batch_size \
        --learning-rate-scaling=sqrt \
        --loss_type $loss_type --glofnd $glofnd \
        --gamma 0.9 \
        --multiprocessing-distributed \
        --world-size 1 --rank 0 --workers 12 \
        --crop-min=.08 \
        --wd=1e-6 \
        --dist-url "tcp://localhost:${port}" \
        --data_name imagenet100 \
        --experiment $experiment \
        --data $dataset \
        --save_dir saved \
        --log_dir $saved_path \
        --run $i \
        --print-freq 100 \
        --alpha $alpha \
        --start_update $start_update \
        --lda_start $lda_start \
        --u_warmup $u_warmup \
        --init_quantile $init_quantile \
        --lr_lda $lr_lda \
        --seed $seed \
        --reset_lambda true \
        && ./scripts/run_lincls.sh $cuda_device $saved_path/$i $seed saved
# done
