#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=glofnd
#SBATCH --output=logs/%x_%j.log

####
# Train model on SLURM cluster with 2 nodes, 2 GPUs each
####

source venv/bin/activate
experiment=tuning

cuda_device=$1
i=$2
dataset="imagenet100"
loss_type=$3
glofnd=$4
alpha=$5
start_update=$6
lda_start=$6
lr_lda=$7
u_warmup=$8
init_quantile=False

timestamp=$(date +"%Y%m%d")
saved_path="${experiment}_${loss_type}_${glofnd}_${alpha}_${start_update}_${lda_start}_${lr_lda}_${u_warmup}"

first_char=${cuda_device:0:1}
port=$((10001 + first_char + lda_start))

# for i in {1..3}; do

    seed=$((12345 + i))
    batch_size=1536

    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr
    export MASTER_PORT=12805
    export CUDA_VISIBLE_DEVICES=$cuda_device
    export WANDB_MODE=disabled

    srun python -u train.py \
        --world-size 4 --workers 6 \
        --lr=.075 --epochs=200 --batch-size=$batch_size \
        --learning-rate-scaling=sqrt \
        --loss_type $loss_type --glofnd $glofnd \
        --gamma 0.9 \
        --crop-min=.08 \
        --wd=1e-6 \
        --data_name imagenet100 \
        --experiment $experiment \
        --data $dataset \
        --save_dir saved \
        --log_dir $saved_path \
        --run_id $i \
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
