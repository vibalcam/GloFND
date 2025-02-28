import os

import random
from sklearn.manifold import TSNE
import torch.nn.functional as F
import numpy as np
# load package
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
from tqdm.auto import tqdm
import argparse
import builtins
from datetime import datetime
import math
import os
import random
import shutil
import wandb
import time
import warnings
import numpy as np
from functools import partial
from utils import utils
# from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models as torchvision_models
# import torchvision.models as torchvision_models
# from torch.utils.tensorboard import SummaryWriter

import sogclr.builder
import sogclr.loader
import sogclr.optimizer
import sogclr.folder # imagenet

torch.cuda.set_device(9)
names = [
    'dcl',
]
model_paths = [
    "baselines/tuning_20241028_153626_dcl_0_0_0_0_-1_0/3/checkpoint_0199.pth.tar",
]
data = "/data/datasets/imagenet100"
arch = "resnet50"
batch_size = 2048
folder_save = "results/embeddings"

# Data loading code
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

image_size = 224
normalize = transforms.Normalize(mean=mean, std=std)

augmentation1 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    normalize,
])

traindir = os.path.join(data, 'train')
train_dataset = sogclr.folder.ImageFolder(
    traindir,
    augmentation1)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False,
    num_workers=6, pin_memory=True, drop_last=False,
    persistent_workers=True)

def _build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)

models = []
for model_path in model_paths:
    model = torchvision_models.__dict__[arch](pretrained=False)
    # Remove the final fully connected layer and add an identity layer
    if hasattr(model, 'fc'):
        linear_keyword = 'fc'
        hidden_dim = model.fc.weight.shape[1]
        # model.fc = nn.Identity()
        model.fc = _build_mlp(2, hidden_dim, 2048, 128)
    else:
        raise ValueError(f"Unsupported model architecture: {arch}")

    # Load checkpoint
    if os.path.isfile(model_path):
        print(f"=> Loading checkpoint '{model_path}'")
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('state_dict', checkpoint)
        lda = state_dict['module.lambda_threshold.lda']
        # get base_encoder state_dict
        state_dict = {k.replace('module.base_encoder.', ''): v for k, v in state_dict.items() if 'module.base_encoder.' in k}
        msg = model.load_state_dict(state_dict, strict=True)
        print(f"=> Loaded checkpoint with missing keys: {msg.missing_keys}")
        # assert all([linear_keyword in k for k in msg.missing_keys]), "Missing keys should be linear layers"
        model.eval()
    else:
        print(f"Warning: Checkpoint {model_path} not found!")
        raise FileNotFoundError

    models.append(model)

from tqdm.auto import tqdm

f_list = []
for name, model in zip(names, models):
    hidden_list1 = []
    indices = []
    meta = []

    with torch.inference_mode():
        model.cuda()

        # tqdm_progress = tqdm(total=len(train_loader), leave=False)
        for images, labels, index in tqdm(train_loader, leave=False, desc=name):
            images = images.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(True):
                hidden1 = model(images)
            hidden1 = F.normalize(hidden1, p=2, dim=1)
            hidden_list1.append(hidden1.cpu())
            indices.append(index.cpu())
            meta.append(labels.cpu())

            # tqdm_progress.update(images.shape[0])

        model.cpu()

    f_name = f"{folder_save}/{name}.pth"
    torch.save(
        {
            "name": name,
            "hidden": torch.cat(hidden_list1, dim=0),
            "indices": torch.cat(indices, dim=0),
            "meta": torch.cat(meta, dim=0),
        }, f_name
    )
    f_list.append(f_name)

del train_loader

seed = 123456
tsne = TSNE(n_components=2, random_state=seed, n_jobs=6, verbose=1)

for f in tqdm(f_list, leave=False, desc="TSNE"):
    data = torch.load(f)
    h = data["hidden"].numpy()
    proj = tsne.fit_transform(h)
    data["proj"] = proj
    data["kl"] = tsne.kl_divergence_
    torch.save(data, f)
