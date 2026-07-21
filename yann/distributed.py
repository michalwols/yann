import os
from collections.abc import Iterable, Iterator, Mapping
from datetime import timedelta
from typing import Any, NamedTuple, Union

import torch
from torch import Tensor, nn
from torch import distributed as dist


class Dist:
  """
  torch.distributed wrapper that also supports non distributed mode
  """

  def __init__(
    self,
    backend='nccl',
    init_method='env://',
    world_size=None,
    rank=None,
  ):
    self.backend = backend
    self.init_method = init_method

    self.world_size = int(
      world_size
      if world_size is not None
      else os.environ.get(
        'WORLD_SIZE',
        torch.cuda.device_count() if torch.cuda.is_available() else 1,
      ),
    )
    self.rank = rank if rank is not None else int(os.environ.get('RANK', 0))
    self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

  @classmethod
  def from_env(cls, backend=None):
    return cls(
      backend=backend or ('nccl' if torch.cuda.is_available() else 'gloo'),
    )

  def initialize(self, timeout=None):
    if not self.is_enabled or not self.is_available() or self.is_initialized():
      return self

    dist.init_process_group(
      backend=self.backend,
      init_method=self.init_method,
      world_size=self.world_size,
      rank=self.rank,
      **({'timeout': timedelta(seconds=timeout)} if timeout is not None else {}),
    )

    if self.backend == 'nccl':
      torch.cuda.set_device(self.local_rank)
    return self

  def cleanup(self):
    dist.destroy_process_group()

  def destroy(self):
    if self.is_available() and self.is_initialized():
      dist.destroy_process_group()

  def is_available(self):
    return dist.is_available()

  def is_initialized(self):
    return dist.is_initialized()

  @property
  def is_enabled(self):
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

  @property
  def enabled(self):
    return self.is_enabled

  @property
  def device(self):
    return f'cuda:{self.local_rank}'

  @property
  def tensor_device(self) -> torch.device:
    return torch.device(
      f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu',
    )

  @property
  def is_main(self):
    return self.rank == 0

  @property
  def main(self):
    return self.is_main

  @property
  def is_active(self):
    return self.is_available() and self.is_initialized() and self.world_size > 1

  def barrier(self):
    if not self.is_available():
      return
    if not self.is_initialized():
      return
    if self.world_size == 1:
      return
    dist.barrier()

  def broadcast_object(self, value: Any, src: int = 0) -> Any:
    if not self.is_active:
      return value
    values = [value]
    dist.broadcast_object_list(values, src=src)
    return values[0]

  def gather_object(self, value: Any, dst: int = 0):
    if not self.is_active:
      return [value]
    output = [None] * self.world_size if self.rank == dst else None
    dist.gather_object(value, output, dst=dst)
    return output

  def reduce(self, value: Tensor, op=None) -> Tensor:
    if self.is_active:
      dist.all_reduce(value, op=op if op is not None else dist.ReduceOp.SUM)
    return value

  def mean(self, value: Union[Tensor, Mapping]):
    if isinstance(value, Mapping):
      return type(value)((key, self.mean(item)) for key, item in value.items())
    output = value.detach().clone()
    self.reduce(output)
    if self.is_active:
      output /= self.world_size
    return output

  def global_mean(self, value_sum, count) -> Tensor:
    pair = torch.tensor(
      [float(value_sum), float(count)],
      device=self.tensor_device,
      dtype=torch.float64,
    )
    self.reduce(pair)
    return pair[0] / pair[1].clamp_min(1)

  def print(self, *args, **kwargs):
    if self.is_main:
      print(*args, **kwargs)

  def __str__(self):
    return f"""Dist(
    backend={self.backend},
    rank={self.rank},
    world_size={self.world_size},
    local_rank={self.local_rank},
    device={self.device},
    pid={os.getpid()}
    )"""


class DistPlacement(NamedTuple):
  rank: Union[int, None] = None
  local_rank: Union[int, None] = None


def matches(placement: Union[int, DistPlacement, None], dist: Dist):
  if placement is None:
    return True
  if isinstance(placement, int):
    return placement == dist.rank
  if isinstance(placement, tuple):
    rank, local_rank = placement
    if rank is not None:
      return rank == dist.rank
    if local_rank is not None:
      return local_rank == dist.local_rank
    return True


def init(backend=None, timeout=None) -> Dist:
  return Dist.from_env(backend).initialize(timeout=timeout)


def wrap(
  model: nn.Module,
  dist: Dist,
  find_unused_parameters: bool = False,
  broadcast_buffers: bool = False,
  static_graph: bool = False,
) -> nn.Module:
  model.to(dist.tensor_device)
  if not dist.is_enabled:
    return model
  return torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[dist.local_rank] if dist.tensor_device.type == 'cuda' else None,
    output_device=dist.local_rank if dist.tensor_device.type == 'cuda' else None,
    find_unused_parameters=find_unused_parameters,
    broadcast_buffers=broadcast_buffers,
    static_graph=static_graph,
  )


def shard(iterable: Iterable, rank: int, world_size: int) -> Iterator:
  for index, item in enumerate(iterable):
    if index % world_size == rank:
      yield item
