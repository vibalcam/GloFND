#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
##SBATCH --wait-all-nodes=1
#SBATCH --job-name=lincls
#SBATCH --output=logs/%x_%j.log

####
# Run linear evaluation on SLURM cluster with 1 node, 1 GPU
####

source venv/bin/activate

dataset="imagenet100"
cuda_device=$1
first_char=${cuda_device:0:1}
port=$((10101 + first_char))

# use sqrt scaling for epochs
percentages=(1 0.1 0.01 0.001)
batchSizes=(1024 1024 1024 1024)
n_epochs=(90 285 900 900)

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805
export CUDA_VISIBLE_DEVICES=$cuda_device
export WANDB_MODE=disabled

length=${#percentages[@]}
for (( i=0; i<$length; i++ )); do
    sample_images=${percentages[$i]}
    epochs=${n_epochs[$i]}
    batch=${batchSizes[$i]}

    echo "Running on GPU $cuda_device with sample_images $sample_images"

    srun python -u lincls.py \
        --world_size 1 --workers 12 \
        -b $batch --lr 0.1 --epochs $epochs \
        --data_name imagenet1000 \
        --data $dataset \
        --save_dir lincls_saved \
        --log_dir $2 \
        --pretrained $4/$2/checkpoint_0199.pth.tar \
        --seed $3 \
        --sample_images $sample_images \
        || { echo "Failed training"; exit 1; }
done
