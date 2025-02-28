#!/bin/bash
#SBATCH --time=1-04:00:00
#SBATCH --mem=120G
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=glofnd
#SBATCH --partition=gpu
#SBATCH --output=train_logs/glofnd_%x_%j.log

source ~/.bashrc

###### Evaluation ######
echo "Starting evaluation..."

###### Params for grace ######
source venv/bin/activate
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export HUGGINGFACE_HUB_CACHE='checkpoints/huggingface'
num_gpus=$SLURM_NTASKS
num_nodes=$SLURM_NNODES

###### Params for both ######
export WANDB_MODE="disabled"
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805
export PYTHONPATH="$PYTHONPATH:$PWD/src"

num_gpus_per_node=$((num_gpus / num_nodes))
if [ "$num_gpus_per_node" -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES='0,1'
elif [ "$num_gpus_per_node" -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES='0,1,2,3'
else
    echo "Job is running on $num_gpus_per_node nodes (unexpected number)."
    exit 1
fi

## Default params
data=cc3m
i=1
glofnd=glofnd
loss_type=v1
alpha=1e-4
start_update=15
lda_start=$start_update
lr_lda=0.05
clip_grad_mult=0.0
init_quantile=False
glofnd_u_warmup=0

## Modify
# pretrained_config=
data=$1; shift 1
i=$1; shift 1
loss_type=$1; shift 1
glofnd=$1; shift 1
alpha=$1; shift 1
start_update=$1; shift 1
lda_start=$start_update
lr_lda=$1; shift 1
clip_grad_mult=$1; shift 1
glofnd_u_warmup=$1; shift 1

checkpoint_id=$((lda_start - 1))
checkpoint_id=$(printf "%04d" $checkpoint_id)

## Set training
name=${data}_${glofnd}_${loss_type}_lr${lr_lda}_alpha${alpha}_su${start_update}_lsu${lda_start}_clip${clip_grad_mult}
logs=saved/$i
mkdir -p $logs

case $data in
    "cc3m")
        train_data='datasets/cc3m_webdataset/cc3m_train/{00000..00331}.tar'
        train_num_samples=2723840 
        data_size=3318333
        batch_size=$((1024 / num_gpus))
        epochs=37
        model=RN50
        lr=1e-3
        rho_v3=6.5
        lr_tau_v3=2e-4
        gamma_decay_epochs=18
        gamma_v2=0.2
        ;;
    *)
        echo "Dataset not recognized: $data"
        exit 1
        ;;
esac
echo "Batch size is $batch_size using $num_gpus GPUs and $num_nodes nodes"

case $loss_type in
    "v1")
        srun python -u src/training/main.py \
            --save-frequency 1 \
            --train-data $train_data \
            --train-num-samples $train_num_samples --data_size $data_size \
            --warmup 10000 \
            --batch-size $batch_size \
            --epochs $epochs \
            --workers 6 \
            --model $model \
            --name ${name} --logs $logs \
            --seed 2024 \
            --profile \
            --wd 0.1 \
            --local-loss \
            --fastclip --temperature_scheme global_constant --temperature 0.03 \
            --lr $lr \
            --gamma 0.2 --gamma_schedule cosine --gamma_decay_epochs $gamma_decay_epochs \
            --report-to "wandb" --wandb-project-name "glofnd-clip" \
            --glofnd $glofnd --glofnd_reset_lda \
            --glofnd_alpha $alpha --glofnd_lr_lda $lr_lda \
            --glofnd_start_update $start_update --glofnd_start_lda $lda_start \
            --glofnd_clip_grad_mult $clip_grad_mult --glofnd_u_warmup $glofnd_u_warmup 
        ;;
    "v3")
        srun python -u src/training/main.py \
            --save-frequency 1 \
            --train-data $train_data \
            --train-num-samples $train_num_samples --data_size $data_size \
            --warmup 10000 \
            --batch-size $batch_size \
            --epochs $epochs \
            --workers 6 \
            --model $model \
            --name ${name} --logs $logs \
            --seed 2024 \
            --profile \
            --wd 0.1 \
            --local-loss \
            --fastclip --multiply_tau --temperature_scheme global_learnable --temperature 0.07 \
            --lr $lr --lr_tau $lr_tau_v3 --lr_tau_scheduler step_thresh --rho $rho_v3 \
            --gamma 0.2 --gamma_schedule cosine --gamma_decay_epochs $gamma_decay_epochs \
            --report-to "wandb" --wandb-project-name "glofnd-clip" \
            --glofnd $glofnd --glofnd_reset_lda \
            --glofnd_alpha $alpha --glofnd_lr_lda $lr_lda \
            --glofnd_start_update $start_update --glofnd_start_lda $lda_start \
            --glofnd_clip_grad_mult $clip_grad_mult --glofnd_u_warmup $glofnd_u_warmup 
        ;;
    *)
        exit 1
        ;;
esac

    echo "Running $name evaluate"
