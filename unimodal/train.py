#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
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
import torchvision.models as torchvision_models
# from torch.utils.tensorboard import SummaryWriter

import sogclr.builder
import sogclr.loader
import sogclr.optimizer
import sogclr.folder # imagenet

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = torchvision_model_names

parser = argparse.ArgumentParser(description='SogCLR ImageNet Pre-Training')
parser.add_argument('--data', metavar='DIR', default='/data/imagenet100/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--dim', default=128, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--mlp-dim', default=2048, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--t', default=0.1, type=float,
                    help='softmax temperature (default: 1.0)')
parser.add_argument('--num_proj_layers', default=2, type=int,
                    help='number of non-linear projection heads')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

# dataset 
parser.add_argument('--data_name', default='imagenet1000', type=str) 
parser.add_argument('--save_dir', default='./saved_models/', type=str) 


# sogclr
parser.add_argument('--loss_type', default='dcl', type=str, choices=['dcl', 'cl'],
                    help='loss function to use (default: dcl)')
parser.add_argument('--glofnd', default='none', type=str, choices=['none', 'glofnd', 'fnd', 'oracle', 'uglofnd', 'disabled'], help='glofnd module to use')
parser.add_argument('--gamma', default=0.9, type=float,
                    help='for updating moving average estimator u for sogclr')
parser.add_argument('--learning-rate-scaling', default='sqrt', type=str,
                    choices=['sqrt', 'linear'],
                    help='learing rate scaling (default: sqrt)')

# CUSTOM ARGUMENTS 
parser.add_argument('--reset_lambda', default=False, type=str2bool, help='reset lambda')
parser.add_argument('--alpha', default=[0.01], type=float, nargs="+", help='alpha for glofnd')
parser.add_argument('--start_update', default=0, type=int, help='start to update lambda')
parser.add_argument('--u_warmup', default=0, type=int, help='update lambda with unique')
parser.add_argument('--lda_start', default=20, type=int, help='start to use lda')
parser.add_argument('--init_quantile', default=False, type=str2bool, help='init quantile')
parser.add_argument('--lr_lda', default=1.0, type=float, help='lda learning rate')
parser.add_argument('--momentum_lda', default=0.95, type=float, help='lda momentum')
parser.add_argument('--log_dir', default=None, type=str, help='log directory')
parser.add_argument('--run_id', default=0, type=int, help='run number')
parser.add_argument('--experiment', default='', type=str, help='experiment name')


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

def load_pretrained(model_path, model):
    # Remove the final fully connected layer and add an identity layer
    if hasattr(model, 'fc'):
        linear_keyword = 'fc'
        hidden_dim = model.fc.weight.shape[1]
        # model.fc = nn.Identity()
        model.fc = _build_mlp(2, hidden_dim, 2048, 128)
    else:
        raise ValueError(f"Unsupported model architecture")

    # Load checkpoint
    if os.path.isfile(model_path):
        print(f"=> Loading checkpoint '{model_path}'")
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get('state_dict', checkpoint)
        # get base_encoder state_dict
        state_dict = {k.replace('module.base_encoder.', ''): v for k, v in state_dict.items() if 'module.base_encoder.' in k}
        msg = model.load_state_dict(state_dict, strict=True)
        print(f"=> Loaded checkpoint with missing keys: {msg.missing_keys}")
        # assert all([linear_keyword in k for k in msg.missing_keys]), "Missing keys should be linear layers"
        model.eval()
    else:
        print(f"Warning: Checkpoint {model_path} not found!")
        raise FileNotFoundError
    
    return model


def log_dict(prefix, data, **kwargs):
    for key, value in data.items():
        wandb.log({f"{prefix}/{key}": value}, **kwargs)


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    # if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        args.world_size = 1
        args.rank = 0  # global rank
        args.local_rank = 0
        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url)
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()

        if torch.cuda.is_available():
            if args.distributed:
                device = 'cuda:%d' % args.local_rank
            torch.cuda.set_device(device)
            # device = torch.device(device)
            args.gpu = device
        
    # sizes for each dataset
    if args.data_name == 'imagenet100': 
        data_size = 129395+1
    elif args.data_name == 'imagenet1000': 
        data_size = 1281167+1 
    else:
        data_size = 1000000 
    print ('pretraining on %s'%args.data_name)

    # create model
    set_all_seeds(2022)
    print("=> creating model '{}'".format(args.arch))
    model = sogclr.builder.SimCLR_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True), 
            args.dim, args.mlp_dim, args.t, loss_type=args.loss_type, glofnd_type = args.glofnd, N=data_size, num_proj_layers=args.num_proj_layers, alpha=args.alpha, lr_lda=args.lr_lda, lda_start=args.lda_start, init_quantile=args.init_quantile, distributed=args.distributed, args=args,)

    # infer learning rate before changing batch size
    if args.learning_rate_scaling == 'linear':
        # infer learning rate before changing batch size
        args.lr = args.lr * args.batch_size / 256
    else:
        # sqrt scaling  
        args.lr = args.lr * math.sqrt(args.batch_size)
        
    print ('initial learning rate:', args.lr)      
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    #print(model) # print model after SyncBatchNorm

    if args.optimizer == 'lars':
        optimizer = sogclr.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
        
    scaler = torch.cuda.amp.GradScaler()

    # log_dir 
    save_root_path = args.save_dir
    global_batch_size = args.batch_size*args.world_size
    method_name = {
        'dcl': 'sogclr',
        'cl': 'simclr',
    }[args.loss_type]
    method_name += '_%s'%args.glofnd
    now = datetime.now()
    date_str = now.strftime('%Y%m%d')
    timestamp_str = now.strftime('%H%M%S')
    if args.log_dir is not None:
        logname = args.log_dir
    else:
        logname = '%s_%s_%s_%s_%s_%s-%s-%s_bz_%s_E%s_WR%s_lr_%.3f_%s_wd_%s_t_%s_g_%s_%s_%s'%(date_str, timestamp_str, args.data_name, args.loss_type, args.arch, method_name, args.dim, args.mlp_dim, global_batch_size, args.epochs, args.warmup_epochs, args.lr, args.learning_rate_scaling, args.weight_decay, args.t, args.gamma, args.optimizer, str(args.alpha) )
        args.log_dir = logname
    logdir = os.path.join(logname, str(args.run_id))
    os.makedirs(os.path.join(save_root_path, logdir), exist_ok=True)
    utils.save_json(vars(args), os.path.join(save_root_path, logdir, 'config.json'))

    # summary_writer = SummaryWriter(log_dir=os.path.join(save_root_path, logdir))
    summary_writer = None
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="glofnd",
        # track hyperparameters and run metadata
        config=args,
        tags=[args.experiment],
        job_type='pretrain',
        name=logname,
        save_code=True,
        group=logname,
    )
    print (logdir)
    
    # optionally resume from a checkpoint
    if args.resume == 'none':
        args.resume = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                # loc = 'cuda:{}'.format(args.gpu)
                loc = args.gpu
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            if args.reset_lambda:
                checkpoint['state_dict'] = {key: value for key, value in checkpoint['state_dict'].items() if "lambda_threshold" not in key}
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            ckp_u = checkpoint['u'].cpu()
            if ckp_u.ndim == 2:
                ckp_u = ckp_u.unsqueeze(0)
            else:
                assert ckp_u.ndim == 3
            if ckp_u.shape[0] == model.module.u.shape[0]:
                model.module.u = ckp_u
            else:
                assert ckp_u.shape[0] == 1
                model.module.u = ckp_u.repeat(model.module.u.shape[0], 1, 1)
            print('check sum u:', model.module.u.sum())
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise ValueError('no checkpoint found at %s'%args.resume)

    cudnn.benchmark = True
    

    # Data loading code
    mean = {'imagenet100':  [0.485, 0.456, 0.406],
            'imagenet1000': [0.485, 0.456, 0.406],
            }[args.data_name]
    std = {'imagenet100':   [0.229, 0.224, 0.225],
            'imagenet1000': [0.229, 0.224, 0.225],
            }[args.data_name]

    image_size = {'imagenet100':224, 'imagenet1000':224}[args.data_name]
    normalize = transforms.Normalize(mean=mean, std=std)

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    # simclr
    augmentation1 = [
        transforms.RandomResizedCrop(image_size, scale=(args.crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([sogclr.loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    if args.data_name == 'imagenet1000' or args.data_name == 'imagenet100' :
        traindir = os.path.join(args.data, 'train')
        train_dataset = sogclr.folder.ImageFolder(
            traindir,
            sogclr.loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                                          transforms.Compose(augmentation1)))
        
        # valdir = os.path.join(args.data, 'val')
        # val_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(valdir, transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize,
        #     ])),
        #     batch_size=256, shuffle=False,
        #     num_workers=args.workers, pin_memory=True
        # )
    else:
        raise ValueError

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        start_time = time.time()
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)
        if args.distributed:
            model.module.dist_update_params()
        else:
            model.dist_update_params()
        print('elapsed time (s): %.1f'%(time.time() - start_time))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0): # only the first GPU saves checkpoint
            
            if (epoch+1) % 10 == 0 or args.epochs - epoch < 3:
                local_u = model.module.u
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'u': model.module.u, 
                    'args': vars(args),
                }, is_best=False, filename=os.path.join(save_root_path, logdir, 'checkpoint_%04d.pth.tar' % epoch) )

    # if args.rank == 0:
    #     summary_writer.close()
    wandb.finish()

def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)

    for i, (images, labels, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss, log_d = model(images[0], images[1], index, gamma=args.gamma, epoch=epoch, class_labels=labels)

        losses.update(loss.item(), images[0].size(0))
        if args.rank == 0:
            # summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)
            wandb.log({"train/loss": loss.item(), "train/epoch": epoch}, step=epoch * iters_per_epoch + i)
            log_dict('train', log_d, step=epoch * iters_per_epoch + i)
        del log_d

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        t = time.time() - end
        batch_time.update(t)
        if args.rank == 0:
            wandb.log({"train/batch_time": t}, step=epoch * iters_per_epoch + i)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    # if args.ref not in [None, 'none']:
    #     if (epoch ) % 100 == 0:
    #         if Path(args.ref).exists():
    #             ref_state_dict = load_pretrained(args.ref, torchvision_models.resnet50(weights=None)).state_dict()
    #         elif args.ref == 'resnet50':
    #             # ref_model = torchvision_models.resnet50(weights='DEFAULT')
    #             ref_state_dict = torch.load("data/resnet50_state_dict.pth", map_location="cpu")
    #         else:
    #             raise ValueError('ref model not supported')
            
    #         # check reference model has not changed
    #         assert model.module.ref is not None, "Reference model is not loaded"
    #         model_state_dict = model.module.ref.state_dict()
    #         for k in ref_state_dict.keys():
    #             assert 'fc' in k or torch.allclose(ref_state_dict[k], model_state_dict[k]), f"Reference model has changed: {k}"


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()
