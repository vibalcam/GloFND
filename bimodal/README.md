# Bimodal GloFND

We show how to use **Glo**bal **F**alse **N**egative **D**etection (*GloFND*) for *global contrastive learning* in the bimodal scenario. For this, we apply GloFND to text-image FastCLIP and SogCLR.

**Table of Contents**
- [Getting Started](#getting-started)
  - [Environment Setup](#environment-setup)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)

## Getting Started

### Environment Setup

1. Download this repository:
    ```bash
    git clone https://github.com/vibalcam/GloFND
    cd bimodal
    ```
2. Create a new environment:
    ```bash
    conda create -n fastclip python=3.11
    conda activate fastclip
    pip install -r requirements-training.txt
    ```

### Training

> Check the `scripts` folder for a complete training script example. The script is designed to run on a SLURM cluster using 4 GPUs (2 nodes and 2 GPUs per node) on the CC3M dataset.

To train on your own data, you need to modify the following options

- `--train-data`: the path to the training data, currently only **webdataset** format is supported.
- `--train-num-samples`: this many samples will be seen for one epoch, we recommend to set it to the actual size of the dataset.
- `--data_size`: the original size of the dataset, this may take a value different from `--train-num-samples`. In the case of CC3M, its metadata contains 3318333 image-URL/caption pairs, but we were only able to down 2723840 of them. So we set `--data_size` to 3318333 and set `--train-num-samples` to 2723848.
- `--epochs`: for this many epochs the model will be trained.
- `--gamma_decay_epochs`: for this many epochs $\gamma$ will decrease from 1.0 to `--gamma`. We recommend to set it to half of `--epochs`.

<details open>
    <summary>Sample script to run <b>GloFND + FastCLIP</b> on CC3M using 4 GPUs (2 nodes and 2 GPUs per node)</summary>

```bash
#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --mem=120G
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=fastclip
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.log

source ~/.bashrc
conda activate fastclip

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805

export CUDA_VISIBLE_DEVICES='0,1'
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

srun python -u src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/cc3m_webdataset/cc3m_train/{00000..00331}.tar' \
    --train-num-samples 2723840 --data_size 3318333 \
    --warmup 10000 \
    --batch-size 128 \
    --epochs 37 \
    --workers 6 \
    --model RN50 \
    --name fastclip \
    --seed 2024 \
    --profile \
    --wd 0.1 \
    --local-loss \
    --fastclip --multiply_tau --temperature_scheme global_learnable \
    --lr 1e-3 --lr_tau 2e-4 --lr_tau_scheduler step_thresh --rho 6.5 \
    --gamma 0.2 --gamma_schedule cosine --gamma_decay_epochs 18 \
    --glofnd glofnd --glofnd_reset_lda \
    --glofnd_alpha 1e-3 --glofnd_lr_lda 0.05 \
    --glofnd_start_update 20 --glofnd_start_lda 20
```
</details>

<details>
    <summary>Sample script to run <b>GloFND + SogCLR</b> on CC3M using 4 GPUs (click to expand):</summary>

Replace the `srun python -u src/training/main.py` command in the FastCLIP script with
```bash
srun python -u src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/cc3m_webdataset/cc3m_train/{00000..00331}.tar' \
    --train-num-samples 2723840 --data_size 3318333 \
    --warmup 10000 \
    --batch-size 128 \
    --epochs 37 \
    --workers 6 \
    --model RN50 \
    --name sogclr \
    --seed 2024 \
    --profile \
    --wd 0.1 \
    --local-loss \
    --fastclip --temperature_scheme global_constant \
    --lr 1e-3 \
    --gamma 0.2 --gamma_schedule cosine --gamma_decay_epochs 18 \
    --glofnd glofnd --glofnd_reset_lda \
    --glofnd_alpha 5e-4 --glofnd_lr_lda 0.05 \
    --glofnd_start_update 15 --glofnd_start_lda 15
```
</details>

**Non-slurm Training**: For non-slurm training, please set `master_addr` manually (e.g., `127.0.0.1`), change `srun python -u src/training/main.py` to `cd src && torchrun --nproc_per_node=2 --rdzv_endpoint=$master_addr -m training.main`, and run the above script with `/bin/bash`.

### Evaluation

**Datacomp**: For evaluation on the Datacomp benchmark, please refer to the "Evaluation" section in the [Datacomp repository](https://github.com/mlfoundations/datacomp?tab=readme-ov-file#evaluation).

**CC3M validation**: To evaluate the model on the CC3M validation set:

<details>
    <summary>Sample script to evaluate on CC3M validation set using 1 GPU </summary>

```bash
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=fastclip
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.log

source ~/.bashrc
conda activate fastclip

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805

export CUDA_VISIBLE_DEVICES='0'
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

srun python -u src/training/main.py \
    --save-frequency 1 \
    --val-data './datasets/cc3m_webdataset/cc3m-validation-{0000..0015}.tar' \
    --data_size 3318333 \
    --warmup 10000 \
    --batch-size 256 \
    --epochs 37 \
    --workers 6 \
    --model RN50 \
    --name fastclip \
    --seed 2024 \
    --profile \
    --wd 0.1 \
    --local-loss \
    --fastclip --multiply_tau --temperature_scheme global_learnable \
    --lr 1e-3 --lr_tau 2e-4 --lr_tau_scheduler step_thresh --rho 6.5 \
    --gamma 0.2 --gamma_schedule cosine --gamma_decay_epochs 18 \
    --glofnd glofnd --glofnd_reset_lda \
    --glofnd_alpha 1e-3 --glofnd_lr_lda 0.05 \
    --glofnd_start_update 20 --glofnd_start_lda 20 \
    --resume checkpoint_to_evaluate.pth.tar
```
</details>

## Acknowledgements

Our implementation is based on [FastCLIP](https://github.com/Optimization-AI/fast_clip)'s repo.
