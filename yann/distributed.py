from torch import distributed as dist
import torch
import os
from typing import NamedTuple, Union


class Dist:
  """
  torch.distributed wrapper that also supports non distributed mode
  """
  def __init__(self, backend='nccl', init_method='env://', world_size=None, rank=None):
    self.backend = backend
    self.init_method = init_method

    self.world_size = int(
        world_size if world_size is not None else
        os.environ.get('WORLD_SIZE', torch.cuda.device_count() if torch.cuda.is_available() else 1)
    )
    self.rank = rank if rank is not None else int(os.environ.get('RANK', 0))
    self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

  def initialize(self):
    if not self.is_enabled or not self.is_available() or self.is_initialized():
      return

    dist.init_process_group(
      backend=self.backend,
      init_method=self.init_method,
      world_size=self.world_size,
      rank=self.rank
    )

    if self.backend == 'nccl':
      torch.cuda.set_device(self.local_rank)

  def cleanup(self):
    dist.destroy_process_group()

  def is_available(self):
    return dist.is_available()

  def is_initialized(self):
    return dist.is_initialized()

  @property
  def is_enabled(self):
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

  @property
  def device(self):
    return f"cuda:{self.local_rank}"

  @property
  def is_main(self):
    return self.rank == 0

  def barrier(self):
    if not self.is_available():
      return
    if not self.is_initialized():
      return
    if self.world_size == 1:
      return
    dist.barrier()

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