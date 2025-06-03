# Unimodal GloFND

We show how to use **Glo**bal **F**alse **N**egative **D**etection (*GloFND*) for *global contrastive learning* in the unimodal scenario. For this we apply GloFND to SogCLR.

**Table of Contents**
- [Getting Started](#getting-started)
  - [Environment Setup](#environment-setup)
  - [Training](#training)
  - [Evaluation](#evaluation)
    - [Linear evaluation](#linear-evaluation)
    - [Transfer learning](#transfer-learning)
- [Acknowledgements](#acknowledgements)

## Getting Started

### Environment Setup

1. Download this repository:
```bash
git clone https://github.com/vibalcam/GloFND
cd unimodal
```
2. Create a new environment:
```bash
conda create -n ml
conda activate ml
pip install -r requirements.txt
```

### Training 

> Check the `scripts` folder for complete training script examples for single GPU, multi-GPU, and SLURM cluster.

ImageNet-100 is a subset with random selected 100 classes from original 1000 classes. To contrust the dataset, please follow these steps:
* Download the train and validation datasets from [ImageNet1K](https://image-net.org/challenges/LSVRC/2012/) website
* Run this [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) to create/move all validation images to each category (class) folder
* Copy images from [train/val.txt](https://github.com/Optimization-AI/SogCLR/blob/main/dataset/ImageNet-S/train.txt) to generate ImageNet-100

We pretrain ResNet-50 for 200 epochs.

<details open>
    <summary>Sample script to run <b>GloFND + SogCLR</b> on ImageNet100 using a single GPU with a batch size of 128</summary>

```bash
#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=glofnd
#SBATCH --output=logs/%x_%j.log

source venv/bin/activate
experiment=tuning

cuda_device="0"
i=1
dataset="imagenet100"
loss_type=dcl
glofnd=glofnd
alpha=0.01
start_update=70
lda_start=70
lr_lda=0.05
u_warmup=0
init_quantile=False

timestamp=$(date +"%Y%m%d")
saved_path="${experiment}_${loss_type}_${glofnd}_${alpha}_${start_update}_${lda_start}_${lr_lda}_${u_warmup}"

first_char=${cuda_device:0:1}
port=$((10001 + first_char + lda_start))

seed=$((12345 + i))
batch_size=128

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805
export CUDA_VISIBLE_DEVICES=$cuda_device
export WANDB_MODE=offline

srun python -u train.py \
    --world-size 1 --workers 12 \
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
    --reset_lambda true
```
</details>

**Non-SLURM Training**: To train locally without SLURM, replace `MASTER_ADDR=localhost`, `MASTER_PORT=$port`, and `srun` with 
```bash
torchrun --nproc_per_node=1 --rdzv-endpoint=localhost:$port --nnodes=1 train.py \
--dist-url "tcp://localhost:${port}" 
```

**Multi-GPU Distributed Training**: For multi-GPU training, replace `world-size 1` with the number of GPUs you want to use, and set `CUDA_VISIBLE_DEVICES` to the GPUs you want to use. For local training, replace as well `nproc_per_node=1` with the number of GPUs you want to use.

**SimCLR Training**: If you want to train SimCLR instead of SogCLR, replace `--loss_type dcl` with `--loss_type cl` and use a batch size of 512.

### Evaluation

#### Linear evaluation

> Check the `scripts` folder for complete linear evaluation script examples for single GPU and SLURM.

We evaluate through linear evaluation by removing the projection head and training a linear classifier on top of the frozen features. We consider a semi-supervised setting where we use different percentages of the training set for training the linear classifier.

<details open>
    <summary>Sample script to run semi-supervised linear evaluation on ImageNet100 using a single GPU</summary>

```bash
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

source venv/bin/activate

dataset="imagenet100"
cuda_device="0"
first_char=${cuda_device:0:1}
port=$((10101 + first_char))
i=1

# use sqrt scaling for epochs
percentages=(1 0.1 0.01 0.001)
batchSizes=(1024 1024 1024 1024)
n_epochs=(90 285 900 900)

seed=$((12345 + i))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805
export CUDA_VISIBLE_DEVICES=$cuda_device
export WANDB_MODE=offline

# Run linear evaluation for each percentage
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
        --log_dir saved \
        --pretrained pretrained_path/checkpoint_0199.pth.tar \
        --seed $seed \
        --sample_images $sample_images
done
```
</details>

#### Transfer learning

To evaluate transfer learning, we train a logistic regression classifier on top of the frozen features.

```bash
python transfer.py \
  --pretrained pretrained_path/checkpoint_0199.pth.tar \
  --name transfer \
  --arch resnet50
```
`results/transfer.ipynb` can be used to check the results.

## Acknowledgements

Our implementation is based on [SogCLR](https://github.com/Optimization-AI/SogCLR)'s repo.
