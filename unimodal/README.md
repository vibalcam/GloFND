# Unimodal GloFND

We show how to use **Glo**bal **F**alse **N**egative **D**etection (*GloFND*) for *global contrastive learning* in the unimodal scenario. For this we apply GloFND to SogCLR.

## Getting Started

### Environment Setup

1. Download this repository:
```bash
git clone https://github.com/vibalcam/combinatorial-cluster-deletion.git
```
2. Create a new environment:
```bash
conda create -n ml
conda activate ml
pip install -r requirements.txt
```

### Training  

ImageNet-100 is a subset with random selected 100 classes from original 1000 classes. To contrust the dataset, please follow these steps:
* Download the train and validation datasets from [ImageNet1K](https://image-net.org/challenges/LSVRC/2012/) website
* Run this [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) to create/move all validation images to each category (class) folder
* Copy images from [train/val.txt](https://github.com/Optimization-AI/SogCLR/blob/main/dataset/ImageNet-S/train.txt) to generate ImageNet-100

We use a batch size of 128 and pretrain ResNet-50 for 200 epochs.
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --lr=.075 --epochs=200 --batch-size=128 \
  --learning-rate-scaling=sqrt \
  --loss_type dcl --glofnd glofnd \
  --gamma 0.9 \
  --multiprocessing-distributed \
  --world-size 1 --rank 0 --workers 12 \
  --crop-min=.08 \
  --wd=1e-6 \
  --dist-url "tcp://localhost:10001" \
  --data_name imagenet100 \
  --experiment pretrain \
  --data $dataset_folder \
  --save_dir saved \
  --log_dir $saved_path \
  --run 1 \
  --print-freq 100 \
  --seed 12346 \
  --alpha 0.01 \
  --start_update 70 \
  --lda_start 70 \
  --u_warmup 0 \
  --init_quantile false \
  --lr_lda 0.05 \
  --reset_lambda true
```

### Linear evaluation

We evaluate through linear evaluation by removing the projection head and training a linear classifier on top of the frozen features. We use a batch size of 1024 and train for 90 epochs.

```bash
CUDA_VISIBLE_DEVICES=0 python lincls.py \
  --dist-url "tcp://localhost:10001" \
  -b 1024 --lr 0.1 --epochs 90 \
  --multiprocessing-distributed --world-size 1 --rank 0 --workers 6 \
  --data_name imagenet100 \
  --data $dataset_folder \
  --save_dir lincls_saved \
  --log_dir $save_path \
  --pretrained $pretrained_path/checkpoint_0199.pth.tar \
  --seed 12346 \
  --sample_images 1
```

### Transfer learning

We evaluate through transfer learning by training a logistic regression classifier on top of the frozen features.

```bash
python transfer.py \
  --pretrained $pretrained_path/checkpoint_0199.pth.tar \
  --name transfer \
  --arch resnet50
```
`results/transfer.ipynb` can be used to check the results.

## Acknowledgements

Our implementation is based on [SogCLR](https://github.com/Optimization-AI/SogCLR)'s repo.
