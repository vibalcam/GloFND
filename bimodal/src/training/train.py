import json
import logging
import math
import os
import time
import contextlib
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed
from fast_clip.log_utils import log_hist_as_image

try:
    import wandb
except ImportError:
    wandb = None

from fast_clip import get_input_dtype
from .distributed import is_master, all_gather_tuple_tensor, ModelReducer, OptimizerReducer
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


@torch.no_grad()
def sync_model(model, ref_model):
    """Sync model weights with a reference model"""
    model = unwrap_model(model)
    ref_model = unwrap_model(ref_model)
    for param, ref_param in zip(model.parameters(), ref_model.parameters()):
        param.data.copy_(ref_param.data.to(param.device))


@torch.no_grad()
def shard_features(
        features: torch.Tensor,
        offset: int,
        save_dir: str | pathlib.Path,
        format: str = "{:08d}.pt",
        num_samples_per_shard: int = 10000,
        ):
    save_dir = pathlib.Path(save_dir)
    assert offset % num_samples_per_shard == 0
    starting_shard = offset // num_samples_per_shard
    num_shards = (features.shape[0] + num_samples_per_shard - 1) // num_samples_per_shard
    logging.info(f"Sharding {features.shape[0]} features into {num_shards} shards")
    for i in range(num_shards):
        shard = features[i * num_samples_per_shard: min((i + 1) * num_samples_per_shard, features.shape[0])].clone()
        shard_path = save_dir / format.format(i + starting_shard)
        torch.save(shard, shard_path)


@torch.no_grad()
def cache_features(ref_model, ref_features_dict, data, args):
    ref_features_offset = args.ref_features_offset
    device = torch.device(args.device)
    ref_model.eval()
    ref_features_all = ref_features_dict["features"]
    is_ref_features_cached = ref_features_dict["is_cached"]

    data['train'].set_epoch(0)
    dataloader = data['train'].dataloader
    num_batches = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        # we assume image indices and text indices are the same
        images, texts, idx, _ = batch
        idx = idx - ref_features_offset
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        data_time_m.update(time.time() - end)

        ref_model_out = ref_model(images, texts)
        ref_features = [ref_model_out["image_features"], ref_model_out["text_features"]]
        ref_features_all[idx, 0] = ref_features[0].to("cpu")
        ref_features_all[idx, 1] = ref_features[1].to("cpu")
        is_ref_features_cached[idx] = True

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if i % args.log_every_n_steps == 0 or batch_count == num_batches:
            batch_size = len(images)
            num_samples = batch_count * batch_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches

            samples_per_second = args.batch_size / batch_time_m.val
            samples_per_second_per_gpu = args.batch_size / batch_time_m.val
            logging.info(
                f"[Rank {args.rank:>2d}] Reference Features Caching: "
                f"[{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, "
                f"{samples_per_second_per_gpu:#g}/s/gpu "
            )

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None,
                    profiler=None, feature_process_group=None, outer_model=None, outer_optimizer=None,
                    ref_model=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    if args.fastclip:
        loss.adjust_hyperparams(epoch)
    if hasattr(model, "process_group"):
        process_group = model.process_group
    else:
        process_group = None
    no_sync_mgr = contextlib.nullcontext
    if not torch.distributed.is_initialized():
        rank = 0
    else:
        rank = torch.distributed.get_rank(feature_process_group)
        if torch.distributed.get_world_size(process_group) == 1 and isinstance(model, torch.nn.parallel.DistributedDataParallel):
            no_sync_mgr = model.no_sync
    offset = rank * args.batch_size

    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    model_reducer, optimizer_reducer = None, None
    if args.comm_freq > 1 or args.comm_freq < 0:
        assert torch.distributed.is_initialized(), "Distributed training is required for comm_freq > 1"
        if "runtime" not in args.reducer:
            if "model" in args.reducer:
                model_reducer = ModelReducer(model)
            if "optimizer" in args.reducer:
                optimizer_reducer = OptimizerReducer(optimizer)

    if outer_model is not None:
        assert outer_optimizer is not None
        sync_model(outer_model, model)

    if ref_model is not None:
        assert args.accum_freq == 1, "Reference model only supported with accum_freq=1"
        ref_model.eval()

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    accel_module = torch.cuda
    current_stream = accel_module.current_stream()

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        if args.stop_iters > 0 and step >= args.stop_iters:
            break

        if not args.skip_scheduler:
            scheduler(step)

        if args.fastclip and ref_model is not None and args.cached_ref_features_dir:
            images, texts, image_idx, text_idx, ref_features = batch
        else:
            images, texts, image_idx, text_idx = batch
            ref_features = None
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        indices = [image_idx, text_idx]
        image_idx = image_idx.unsqueeze(-1).to(device=device, dtype=image_idx.dtype, non_blocking=True)
        text_idx = text_idx.unsqueeze(-1).to(device=device, dtype=text_idx.dtype, non_blocking=True)
        indices_device = [image_idx, text_idx]

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if (args.comm_freq > 1 and i % args.comm_freq == 0) or args.comm_freq < 0:
            if "runtime" in args.reducer:
                if "model" in args.reducer:
                    model_reducer = ModelReducer(model)
                if "optimizer" in args.reducer:
                    optimizer_reducer = OptimizerReducer(optimizer)
            if model_reducer is not None:
                model_reducer.all_reduce(async_op=False)
                model_reducer.update()
                current_stream.wait_stream(model_reducer.get_stream())
            # wait for optimizer to create states
            if step > 0 and optimizer_reducer is not None:
                optimizer_reducer.all_reduce(async_op=False)
                optimizer_reducer.update()
                current_stream.wait_stream(optimizer_reducer.get_stream())
            if "runtime" in args.reducer:
                model_reducer, optimizer_reducer = None, None
            # skip for i = 0 since the gradient is 0
            if i > 0 and outer_model is not None:
                outer_optimizer.zero_grad()
                for param, outer_param in zip(model.parameters(), outer_model.parameters()):
                    outer_param.grad = outer_param - param.detach().to(outer_param.device)
                outer_optimizer.step()
                sync_model(model, outer_model)

        if args.accum_freq == 1:
            if args.fastclip and ref_model is not None:
                if ref_features is None:
                    with torch.no_grad():
                        ref_model_out = ref_model(images, texts)
                        ref_features = [ref_model_out["image_features"], ref_model_out["text_features"]]
                else:
                    # [batch, image/text, feature]
                    ref_features = ref_features.movedim(1, 0)
                ref_features = [ref_features[0].to(device), ref_features[1].to(device)]
                ref_remote_features = all_gather_tuple_tensor(ref_features, feature_process_group)
            else:
                ref_remote_features = None
            with no_sync_mgr():
                with autocast():
                    model_out = model(images, texts)
                    logit_scale = model_out["logit_scale"]
                    if args.distill:
                        with torch.no_grad():
                            dist_model_out = dist_model(images, texts)
                        model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                    if args.fastclip:
                        features = [model_out["image_features"], model_out["text_features"]]
                        remote_features = all_gather_tuple_tensor(features, feature_process_group)
                        local_args = {"features": features, "indices": indices, "remote_features": remote_features,
                                      "logit_scale": logit_scale, "offset": offset, "ref_features": ref_features,
                                      "ref_remote_features": ref_remote_features}
                        if args.glofnd == 'none':
                            loss1_im, loss1_tt, sim_im, sim_tt, gather_list = loss.local(**local_args)
                            log_dict = None
                        else:
                            loss1_im, loss1_tt, sim_im, sim_tt, lda_mask, lda_sum, gather_list, log_dict = loss.local(**local_args)
                        # here sim_im is local_im vs. global_tt, sim_tt is local_tt vs. global_im
                        sim = (sim_tt.T, sim_im.T)
                        u = gather_list[0:2]
                        # sync the whole world
                        remote_gather_list = all_gather_tuple_tensor(gather_list, None)
                        remote_indices = all_gather_tuple_tensor(indices_device, None)
                        remote_indices[0] = remote_indices[0].squeeze(-1).to(device="cpu", dtype=torch.int64)
                        remote_indices[1] = remote_indices[1].squeeze(-1).to(device="cpu", dtype=torch.int64)
                        loss.set_params(*remote_indices, *remote_gather_list)
                        
                        if args.glofnd != 'none':
                            gather_list = gather_list + lda_sum
                            gather_lda_mask = all_gather_tuple_tensor(lda_mask, feature_process_group)

                        remote_gather_list = all_gather_tuple_tensor(gather_list, feature_process_group)
                        remote_indices = all_gather_tuple_tensor(indices_device, feature_process_group)
                        remote_indices[0] = remote_indices[0].squeeze(-1).to(device=device, dtype=torch.int64, non_blocking=True)
                        remote_indices[1] = remote_indices[1].squeeze(-1).to(device=device, dtype=torch.int64, non_blocking=True)
                        remote_u = remote_gather_list[0:2]
                        if "individual" in args.temperature_scheme:
                            remote_tau = remote_gather_list[2:4]
                            remote_bounds = remote_gather_list[4:6]
                            model_out.update({"remote_tau": remote_tau, "remote_bounds": remote_bounds})
                        model_out.update(
                            {"features": features, "remote_features": remote_features, "remote_u": remote_u, "remote_indices": remote_indices})
                        if args.glofnd != 'none':
                            lda_mask = gather_lda_mask
                            lda_sum = remote_gather_list[-2:]
                            model_out.update({"lda_mask": lda_mask, "lda_sum": lda_sum})
                        model_out.update(
                            {"offset": offset, "loss1": (loss1_im, loss1_tt), "u": u, "sim": sim})
                        model_out.update(
                            {"ref_features": ref_features, "ref_remote_features": ref_remote_features})
                    model_out.update({"epoch": epoch, "args": args, "idx_im":image_idx, "idx_tt":text_idx})
                    losses = loss(
                        **model_out, output_dict=True,
                    )

                    loss_log_dict = losses.pop('log_dict', {})
                    dist_log = {}
                    if log_dict is not None:
                        loss_log_dict.update({k:v.mean().detach().cpu().item() for k,v in log_dict.items()})
                        dist_log['num_negatives_perc_im'] = log_dict['num_negatives_perc_im'].detach().cpu().numpy()
                        dist_log['num_negatives_perc_tt'] = log_dict['num_negatives_perc_tt'].detach().cpu().numpy()
                        dist_log['num_filtered_perc_im'] = log_dict['num_filtered_perc_im'].detach().cpu().numpy()
                        dist_log['num_filtered_perc_tt'] = log_dict['num_filtered_perc_tt'].detach().cpu().numpy()
                        del log_dict
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

                # Gradient clipping
                if args.grad_norm_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_norm_clip, norm_type=2.0)
        else:
            raise NotImplementedError("Accumulation frequency > 1 not supported")

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(args.logit_scale_bound))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.batch_size / batch_time_m.val
            if args.fastclip and "individual" in args.temperature_scheme:
                lr_tau = loss.lr_tau
            else:
                lr_tau = optimizer.param_groups[-1]['lr']
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} LR_tau: {lr_tau:.5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})
            log_data.update(loss_log_dict)

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    assert False, "Tensorboard logging not supported"
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})
            
            # log distributions
            for name, val in dist_log.items():
                name = "train/" + name
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    log_hist_as_image(val, f"distributions/{name}")
            del dist_log

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

        if profiler is not None:
            profiler.step()
    # end for

    if args.comm_freq > 1 or args.comm_freq < 0:
        if "runtime" in args.reducer:
            if "model" in args.reducer:
                model_reducer = ModelReducer(model)
            if "optimizer" in args.reducer:
                optimizer_reducer = OptimizerReducer(optimizer)
        if model_reducer is not None:
            model_reducer.all_reduce(async_op=False)
            model_reducer.update()
            current_stream.wait_stream(model_reducer.get_stream())
        if optimizer_reducer is not None:
            optimizer_reducer.all_reduce(async_op=False)
            optimizer_reducer.update()
            current_stream.wait_stream(optimizer_reducer.get_stream())
        # NOTE we release the reducers since they will be recreated in the next epoch
        model_reducer, optimizer_reducer = None, None
        if outer_model is not None:
            outer_optimizer.zero_grad()
            for param, outer_param in zip(model.parameters(), outer_model.parameters()):
                outer_param.grad = outer_param - param.detach().to(outer_param.device)
            outer_optimizer.step()
            sync_model(model, outer_model)


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {"epoch": epoch}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts, return_zetas=False)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.logs, f"results_{args.name}.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
