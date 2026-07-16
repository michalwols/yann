from __future__ import annotations

from typing import Any

import torch


class Multilog:
  """Callback adapter for any logger exposing ``metric(name, value, **attrs)``."""

  def __init__(self, log: Any, *, every: int = 1, prefix: str = 'train'):
    self.log = log
    self.every = every
    self.prefix = prefix

  def on_batch_end(self, *, trainer, output, **kwargs):
    if trainer.steps % self.every:
      return
    if not isinstance(output, dict):
      return
    for name, value in output.items():
      if torch.is_tensor(value) and value.numel() == 1:
        value = value.detach().item()
      if isinstance(value, (int, float)):
        self.log.metric(
          f'{self.prefix}.{name}',
          value,
          step=trainer.steps,
          optim_step=trainer.optim_steps,
        )
