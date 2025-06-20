import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
import functools
import socket
from typing import Optional

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from fast_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss, create_model, get_model_config
from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object, get_sync_group
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown, step_lr_thresh
from training.train import train_one_epoch, evaluate, cache_features, shard_features
from training.file_utils import pt_load, start_sync_process, remote_sync
from training.optimizer import LAMB, Lion


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def create_optimizer(
        optimizer_name: str,
        model: torch.nn.Module,
        lr: float,
        lr_tau: float,
        wd: float,
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        eps: Optional[float] = None,
        momentum: Optional[float] = None,
) -> torch.optim.Optimizer:
    is_logit_scale = lambda n, p: "logit_scale" in n
    exclude = lambda n, p: (p.ndim < 2 and "logit_scale" not in n) or "bn" in n or "ln" in n or "bias" in n
    include = lambda n, p: not exclude(n, p) and not is_logit_scale(n, p)

    if lr_tau < 0.0:
        lr_tau = lr

    logging.info(f"optimizer: {optimizer_name}, lr: {lr}, lr_tau: {lr_tau}")

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = []
    gain_or_bias_params_names = []
    rest_params = []
    rest_params_names = []
    logit_scale_params = []
    logit_scale_params_names = []
    for n, p in named_parameters:
        if exclude(n, p) and p.requires_grad:
            gain_or_bias_params.append(p)
            gain_or_bias_params_names.append(n)
        elif include(n, p) and p.requires_grad:
            rest_params.append(p)
            rest_params_names.append(n)
        elif is_logit_scale(n, p) and p.requires_grad:
            logit_scale_params.append(p)
            logit_scale_params_names.append(n)

    if optimizer_name == "adamw":
        assert beta1 is not None and beta2 is not None and eps is not None
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": wd},
                {"params": logit_scale_params, "weight_decay": 0., "lr": lr_tau},
            ],
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
        )
    elif optimizer_name == "lamb":
        assert beta1 is not None and beta2 is not None and eps is not None
        optimizer = LAMB(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": wd},
                {"params": logit_scale_params, "weight_decay": 0., "lr": lr_tau},
            ],
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
        )
    elif optimizer_name == "lion":
        assert beta1 is not None and beta2 is not None
        optimizer = Lion(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": wd},
                {"params": logit_scale_params, "weight_decay": 0., "lr": lr_tau},
            ],
            lr=lr,
            betas=(beta1, beta2),
        )
    elif optimizer_name == "nesterov" or optimizer_name == "sgd":
        assert momentum is not None
        if optimizer_name == "nesterov":
            nesterov = True
        else:
            nesterov = False
        optimizer = optim.SGD(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": wd},
                # NOTE we set momentum to 0 for logit scale
                {"params": logit_scale_params, "weight_decay": 0., "lr": lr_tau, "momentum": 0.},
            ],
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizer


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest and args.resume is None:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if args.cache_ref_model_features:
        args.ref_features_path = os.path.join(log_base_path, f"reference_features")
    else:
        args.ref_features_path = ""
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path, args.ref_features_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    if args.fastclip:
        model_kwargs['init_logit_scale'] = np.log(1 / args.temperature)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
        **model_kwargs,
    )
    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model,
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from fast_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)
    if args.reduction_strategy == "inner_outer":
        logging.info("Creating outer model for inner-outer reduction strategy.")
        outer_model = create_model(
            args.model,
            args.pretrained,
            precision=args.precision,
            device="cpu",
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            force_custom_text=args.force_custom_text,
            force_patch_dropout=args.force_patch_dropout,
            force_image_size=args.force_image_size,
            pretrained_image=args.pretrained_image,
            **model_kwargs,
        )
    else:
        outer_model = None
    if args.ref_model:
        logging.info("Creating reference model.")
        ref_model, ref_preprocess_train, ref_preprocess_val = create_model_and_transforms(
            args.ref_model,
            args.ref_model_pretrained,
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            force_custom_text=args.force_custom_text,
            force_patch_dropout=args.force_patch_dropout,
            force_image_size=args.force_image_size,
            pretrained_image=args.pretrained_image,
            image_mean=args.image_mean,
            image_std=args.image_std,
            aug_cfg=args.aug_cfg,
            output_dict=True,
            **model_kwargs,
        )
        if not args.ref_model_pretrained and args.ref_model_checkpoint:
            logging.info(f"Loading reference model checkpoint from {args.ref_model_checkpoint}")
            ref_model_checkpoint = pt_load(args.ref_model_checkpoint, map_location='cpu')
            sd = ref_model_checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            ref_model.load_state_dict(sd)
    else:
        ref_model = None
    if args.ref_model and args.cache_ref_model_features:
        preprocess_train, preprocess_val = ref_preprocess_train, ref_preprocess_val

    if "constant" in args.temperature_scheme or "individual" in args.temperature_scheme:
        model.logit_scale.requires_grad = False

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if args.comm_freq > 1 and args.num_sync_workers <= 0:
        raise ValueError("num_sync_workers must be greater than 0 when using comm_freq > 1.")
    if args.distributed:
        logging.info("Initializing group to sync gradients")
        process_group = get_sync_group(args.num_sync_workers)
        if args.num_feat_sync_workers <= 0:
            args.num_feat_sync_workers = args.num_sync_workers
        logging.info("Initializing group to sync features")
        feature_process_group = get_sync_group(args.num_feat_sync_workers)
    else:
        process_group = None
        feature_process_group = None

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {"process_group": process_group}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    outer_optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'
        optimizer = create_optimizer(
            args.optimizer,
            model,
            args.lr,
            args.lr_tau,
            args.wd,
            args.beta1,
            args.beta2,
            args.eps,
            args.momentum,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

        if outer_model is not None:
            assert args.outer_optimizer in ["sgd", "nesterov"]
            outer_optimizer = create_optimizer(
                args.outer_optimizer,
                outer_model,
                args.outer_lr,
                1.0,
                0.0,
                momentum=args.outer_momentum,
            )

    loss = create_loss(args, feature_process_group)

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            elif args.distributed and not next(iter(sd.items()))[0].startswith('module'):
                sd = {f"module.{k}": v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            if args.fastclip:
                loss.u_im = checkpoint["u_im"]
                loss.u_tt = checkpoint["u_tt"]
                if "individual" in args.temperature_scheme:
                    loss.tau_im = checkpoint["tau_im"]
                    loss.tau_tt = checkpoint["tau_tt"]
                    loss.m_grad_tau_im = checkpoint["m_grad_tau_im"]
                    loss.m_grad_tau_tt = checkpoint["m_grad_tau_tt"]
                    loss.v_grad_tau_im = checkpoint["v_grad_tau_im"]
                    loss.v_grad_tau_tt = checkpoint["v_grad_tau_tt"]
                    loss.bound_im = checkpoint["bound_im"]
                    loss.bound_tt = checkpoint["bound_tt"]
            if args.glofnd != 'none' and not args.glofnd_reset_lda:
                loss.lda_im.load_state_dict(checkpoint["lda_im"])
                loss.lda_tt.load_state_dict(checkpoint["lda_tt"])
            if outer_optimizer is not None:
                outer_optimizer.load_state_dict(checkpoint["outer_optimizer"])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))
    assert len(data), 'At least one train or eval dataset must be specified.'

    # load or cache reference model features
    if args.ref_model and args.cache_ref_model_features:
        if os.path.exists(os.path.join(args.ref_features_path, f"{args.ref_model}_features.pt")):
            logging.info("Loading reference model features.")
            ref_features_dict = torch.load(os.path.join(args.ref_features_path, f"{args.ref_model}_features.pt"),
                                           map_location='cpu')
            ref_features_all = ref_features_dict["features"]
            is_ref_features_cached = ref_features_dict["is_cached"]
        else:
            model_cfg = get_model_config(args.ref_model)
            embed_dim = model_cfg["embed_dim"]
            ref_features_all = torch.zeros((args.data_size, 2, embed_dim))
            is_ref_features_cached = torch.zeros(args.data_size, dtype=torch.bool)
    else:
        ref_features_all = None
        is_ref_features_cached = None
    ref_features_dict = {"features": ref_features_all, "is_cached": is_ref_features_cached}

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        num_batches = data["train"].dataloader.num_batches
        if args.stop_iters <= 0:
            total_steps = num_batches * args.epochs
        else:
            total_steps = args.stop_iters
        lr_list = [args.lr, args.lr]
        lr_tau_list = [args.lr_tau]
        param_groups = optimizer.param_groups
        logit_scale_group = param_groups[-1:]
        other_groups = param_groups[:-1]
        sche_list = []
        # NOTE for const and step, we set warmup to 0
        for lr_scheduler, group, lr_l in zip([args.lr_scheduler, args.lr_tau_scheduler], [other_groups, logit_scale_group], [lr_list, lr_tau_list]):
            if lr_scheduler == "cosine":
                sche = cosine_lr(group, lr_l, args.warmup, total_steps, args.lr_min)
            elif lr_scheduler == "const":
                sche = const_lr(group, lr_l, 0, total_steps)
            elif lr_scheduler == "const-cooldown":
                assert args.epochs_cooldown is not None,\
                    "Please specify the number of cooldown epochs for this lr schedule."
                cooldown_steps = num_batches * args.epochs_cooldown
                sche = const_lr_cooldown(
                    group, lr_l, args.warmup, total_steps,
                    cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
            elif lr_scheduler == "step_thresh":
                sche = step_lr_thresh(group, lr_l, 0, [0.03], [1/3], model=model if not hasattr(model, "module") else model.module)
            else:
                logging.error(
                    f'Unknown scheduler, {lr_scheduler}. Available options are: cosine, const, const-cooldown, step_thresh.')
                exit(1)
            sche_list.append(sche)
        def _lr_adjuster(step, sche_list):
            for sche in sche_list:
                sche(step)
        scheduler = functools.partial(_lr_adjuster, sche_list=sche_list)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        # wandb.require('core')
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        # wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from fast_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        evaluate(model, data, start_epoch, args, writer)
        return

    # profiler
    tb_trace_handler = torch.profiler.tensorboard_trace_handler(os.path.join(args.logs, args.name, "traces"),
                                                                f"{args.rank:02d}_{args.name}_{socket.gethostname()}")

    def trace_handler(prof: torch.profiler.profile, extra_handler):
        for handler in extra_handler:
            handler(prof)
        # prof.export_stacks(os.path.join(args.logs, args.name, "traces", "stacks_cpu.txt"), "self_cpu_time_total")
        # prof.export_stacks(os.path.join(args.logs, args.name, "traces", "stacks_cuda.txt"), "self_cuda_time_total")

    if args.profile:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=10, warmup=2, active=10, repeat=1),
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=functools.partial(trace_handler, extra_handler=[tb_trace_handler]),
            record_shapes=True,
            with_stack=False,
            profile_memory=True,
            # with_stack=True,
            # experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        )
    else:
        prof = None

    # disable logging model arch and hyperparams for evaluation jobs
    if is_master(args) and args.train_data:
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.cache_ref_model_features:
        del model
        logging.info(f'Start caching reference model features.')
        cache_features(ref_model, ref_features_dict, data, args)
        logging.info(f'Finish caching reference model features.')
        if args.cached_ref_features_dir:
            os.makedirs(args.cached_ref_features_dir, exist_ok=True)
            shard_features(ref_features_dict["features"], args.ref_features_offset,
                           args.cached_ref_features_dir, num_samples_per_shard=args.num_samples_per_shard)
            torch.save(ref_features_dict["is_cached"], os.path.join(args.ref_features_path, f"{args.ref_model}_is_cached_{args.rank}.pt"))
        else:
            torch.save(ref_features_dict, os.path.join(args.ref_features_path, f"{args.ref_model}_features_{args.rank}.pt"))
    else:
        for epoch in range(start_epoch, args.epochs):
            if args.stop_epochs > 0 and epoch >= args.stop_epochs:
                logging.info(f'Stopping training at epoch {epoch}.')
                break
            if args.stop_iters > 0 and epoch * data['train'].dataloader.num_batches >= args.stop_iters:
                logging.info(f'Stopping training at iteration {args.stop_iters}.')
                break
            if is_master(args):
                logging.info(f'Start epoch {epoch}')

            train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args,
                            tb_writer=writer, profiler=prof, feature_process_group=feature_process_group,
                            outer_model=outer_model, outer_optimizer=outer_optimizer, ref_model=ref_model)
            completed_epoch = epoch + 1

            if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
                evaluate(model, data, completed_epoch, args, writer)

            # Saving checkpoints.
            if args.save_logs:
                checkpoint_dict = {
                    "epoch": completed_epoch,
                    "name": args.name,
                    "state_dict": original_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if scaler is not None:
                    checkpoint_dict["scaler"] = scaler.state_dict()
                if args.fastclip:
                    checkpoint_dict["u_im"] = loss.u_im
                    checkpoint_dict["u_tt"] = loss.u_tt
                    if "individual" in args.temperature_scheme:
                        tau_dict = {"tau_im": loss.tau_im, "tau_tt": loss.tau_tt,
                                    "m_grad_tau_im": loss.m_grad_tau_im, "m_grad_tau_tt": loss.m_grad_tau_tt,
                                    "v_grad_tau_im": loss.v_grad_tau_im, "v_grad_tau_tt": loss.v_grad_tau_tt,
                                    "bound_im": loss.bound_im, "bound_tt": loss.bound_tt}
                        checkpoint_dict.update(tau_dict)
                if args.glofnd != 'none':
                    checkpoint_dict["lda_im"] = loss.lda_im.state_dict()
                    checkpoint_dict["lda_tt"] = loss.lda_tt.state_dict()
                if outer_optimizer is not None:
                    checkpoint_dict["outer_optimizer"] = outer_optimizer.state_dict()

                if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
                ):
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                    )
                if args.delete_previous_checkpoint:
                    previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                    if os.path.exists(previous_checkpoint):
                        os.remove(previous_checkpoint)

                if args.save_most_recent:
                    # try not to corrupt the latest checkpoint if save fails
                    tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                    latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                    torch.save(checkpoint_dict, tmp_save_path)
                    os.replace(tmp_save_path, latest_save_path)

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')

    logging.info('Finished training.')

    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
