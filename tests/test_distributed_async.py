import asyncio

import torch
from torch import nn

from yann import distributed
from yann.asyncio import amap


def test_single_process_dist_and_shard():
  dist = distributed.Dist.from_env()
  assert not dist.enabled
  assert dist.main
  model = nn.Linear(2, 2)
  assert distributed.wrap(model, dist) is model
  assert list(distributed.shard(range(8), rank=1, world_size=3)) == [1, 4, 7]
  value = dist.global_mean(torch.tensor(6.0), 3)
  assert value.item() == 2.0


def test_async_map():
  async def square(value):
    await asyncio.sleep(0)
    return value * value

  assert asyncio.run(amap(square, range(5), concurrency=2)) == [0, 1, 4, 9, 16]
