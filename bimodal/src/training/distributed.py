import os
import re
import logging
from typing import List, Union, Optional

import torch
import torch.distributed as dist

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_horovod():
    # NOTE w/ horovod run, OMPI vars should be set, but w/ SLURM PMI vars will be set
    # Differentiating between horovod and DDP use via SLURM may not be possible, so horovod arg still required...
    ompi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    pmi_vars = ["PMI_RANK", "PMI_SIZE"]
    if all([var in os.environ for var in ompi_vars]) or all([var in os.environ for var in pmi_vars]):
        return True
    else:
        return False


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


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


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if args.horovod:
        assert hvd is not None, "Horovod is not installed"
        hvd.init()
        args.local_rank = int(hvd.local_rank())
        args.rank = hvd.rank()
        args.world_size = hvd.size()
        args.distributed = True
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif is_using_distributed():
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
        args.distributed = True

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)
    return device


def broadcast_object(args, obj, src=0):
    # broadcast a pickle-able python object from rank-0 to all ranks
    if args.horovod:
        return hvd.broadcast_object(obj, root_rank=src)
    else:
        if args.rank == src:
            objects = [obj]
        else:
            objects = [None]
        dist.broadcast_object_list(objects, src=src)
        return objects[0]


def all_gather_object(args, obj, dst=0):
    # gather a pickle-able python object across all ranks
    if args.horovod:
        return hvd.allgather_object(obj)
    else:
        objects = [None for _ in range(args.world_size)]
        dist.all_gather_object(objects, obj)
        return objects


def all_gather_tuple_tensor(tensor_list: List[torch.Tensor],
                            group: Optional[dist.ProcessGroup] = None,
                            ):
    """all gather m tensors, each of shape n * d, using one all_gather operation"""
    if not dist.is_initialized():
        return tensor_list
    world_size = dist.get_world_size(group)
    tensor = torch.stack(tensor_list)                                               # shape [m*n*d]
    gathered_tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensor_list, tensor, group=group)
    gathered_tensor = torch.stack(gathered_tensor_list).permute((1, 0, 2, 3))       # shape [m*k*n*d]
    return_tensor_list = []
    for i in range(gathered_tensor.shape[0]):
        return_tensor_list.append(gathered_tensor[i].flatten(end_dim=1))
    return return_tensor_list


def get_sync_group(num_sync_workers: int):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if num_sync_workers <= 0:
        group = dist.group.WORLD
    else:
        assert world_size % num_sync_workers == 0
        for i in range(world_size // num_sync_workers):
            group_ = dist.new_group(list(range(i * num_sync_workers, (i + 1) * num_sync_workers)))
            if i == rank // num_sync_workers:
                group = group_
    logging.info(f"Group size: {dist.get_world_size(group)}")
    return group


accel_module = torch.cuda


class Bucket:
    def __init__(self):
        self.data = torch.empty([])
        self.variables = []
        self.offsets = []
        self.lengths = []
        self.views = []
        self.names = []
        self.is_initialized = False

    def init(self,
             variables,
             with_name=False,
             name_filters=[],
             filter_no_grad=False,
             ):
        offset = 0
        dtype = None
        device = None
        name = ""
        for var in variables:
            if with_name:
                name, var = var
                if any(re.fullmatch(name_filter, name) for name_filter in name_filters):
                    continue
            if filter_no_grad and not var.requires_grad:
                continue
            self.names.append(name)
            if dtype is None:
                dtype = var.dtype
            if device is None:
                device = var.device
            self.variables.append(var)
            self.offsets.append(offset)
            self.lengths.append(var.numel())
            offset += var.numel()
        self.data = torch.empty(offset, dtype=dtype, device=device)
        for var, offset, length in zip(self.variables, self.offsets, self.lengths):
            self.views.append(self.data[offset:offset+length].view_as(var))
        self.is_initialized = True

    def copy_variables_to_views(self):
        for var, view in zip(self.variables, self.views):
            view.data.copy_(var.data)

    def copy_views_to_variables(self):
        for var, view in zip(self.variables, self.views):
            var.data.copy_(view.data)


class Reducer:
    def __init__(self,
                 ref: Union[torch.nn.Module, torch.optim.Optimizer],
                 group: Optional[dist.ProcessGroup] = None,
                 stream: Optional[accel_module.Stream] = None,
                 ):
        self._ref = ref
        self._copy = Bucket()
        self._world_size = dist.get_world_size(group)
        if stream is None:
            self._stream = accel_module.current_stream()
        else:
            self._stream = stream
        self._work_list = []
        self._is_reduced = False

    def _build_copy(self):
        raise NotImplementedError

    def all_reduce(self,
                   async_op: bool = True,
                   ):
        with accel_module.stream(self._stream):
            if not self._copy.is_initialized:
                self._build_copy()
            self._copy.copy_variables_to_views()
            work = dist.all_reduce(self._copy.data, async_op=True)
            self._work_list.append(work)
            if not async_op:
                self.wait()

    def update(self,
               update_ref: bool = True,
               ):
        with accel_module.stream(self._stream):
            self.wait()
            if self._is_reduced:
                self._copy.data.mul_(1.0 / self._world_size)
                if update_ref:
                    self._copy.copy_views_to_variables()
            self._is_reduced = False

    def get_stream(self):
        return self._stream

    def wait(self):
        for work in self._work_list:
            work.wait()
        if len(self._work_list) > 0:
            self._is_reduced = True
        self._work_list = []


class ModelReducer(Reducer):
    def __init__(self,
                 ref: Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel],
                 group: Optional[dist.ProcessGroup] = None,
                 stream: Optional[accel_module.Stream] = None,
                 filter: str = "",
                 ):
        super().__init__(ref, group, stream)
        self.filter = filter
        if isinstance(ref, torch.nn.parallel.DistributedDataParallel):
            self._ref = ref.module

    def _build_copy(self):
        name_filters = []
        filter_no_grad = False
        if "bn" in self.filter:
            for name, module in self._ref.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    name_filters.append(name)
            name_filters.append(".*\\.bn\\d+\\..*")
        if "grad" in self.filter:
            filter_no_grad = True
        self._copy.init(self._ref.named_parameters(), with_name=True, name_filters=name_filters,
                        filter_no_grad=filter_no_grad)


class OptimizerReducer(Reducer):
    def __init__(self,
                 ref: torch.optim.Optimizer,
                 group: Optional[dist.ProcessGroup] = None,
                 stream: Optional[accel_module.Stream] = None,
                 ):
        super().__init__(ref, group, stream)

    def _build_copy(self):
        states = []
        for group in self._ref.param_groups:
            for p in group['params']:
                state = self._ref.state[p]
                for key, val in state.items():
                    if val is None or val.device == torch.device("cpu"):
                        continue
                    states.append(val)
        self._copy.init(states)
