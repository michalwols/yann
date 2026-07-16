from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from yann.callbacks.base import Callback


class Multilog(Callback):
  """Callback adapter for any logger exposing ``metric(name, value, **attrs)``."""

  def __init__(self, log: Any, every: int = 1, prefix: str = 'train'):
    self.log = log
    self.every = every
    self.prefix = prefix

  def on_step_end(
    self,
    index=None,
    inputs=None,
    targets=None,
    outputs=None,
    loss=None,
    trainer=None,
  ):
    if trainer is None or trainer.num_steps % self.every:
      return
    metrics = dict(outputs) if isinstance(outputs, Mapping) else {}
    if loss is not None:
      metrics.setdefault('loss', loss)
    for name, value in metrics.items():
      if torch.is_tensor(value) and value.numel() == 1:
        value = value.detach().item()
      if isinstance(value, (int, float)):
        self.log.metric(
          f'{self.prefix}.{name}',
          value,
          step=trainer.num_steps,
          optim_step=trainer.num_optim_steps,
        )
