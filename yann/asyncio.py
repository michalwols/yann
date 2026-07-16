from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any


async def gather_limited(
  awaitables: Iterable[Awaitable[Any]],
  *,
  concurrency: int = 32,
):
  semaphore = asyncio.Semaphore(concurrency)

  async def run(awaitable):
    async with semaphore:
      return await awaitable

  return await asyncio.gather(*(run(awaitable) for awaitable in awaitables))


async def amap(
  fn: Callable[[Any], Awaitable[Any]],
  items: Iterable[Any],
  *,
  concurrency: int = 32,
):
  return await gather_limited((fn(item) for item in items), concurrency=concurrency)
