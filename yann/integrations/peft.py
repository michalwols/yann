from __future__ import annotations

from pathlib import Path
from typing import Any


def lora(
  model,
  *,
  rank: int = 16,
  alpha: int | None = None,
  dropout: float = 0.0,
  targets: str | list[str] = 'all-linear',
  task_type: str = 'CAUSAL_LM',
  **kwargs: Any,
):
  try:
    from peft import LoraConfig, get_peft_model
  except ImportError as error:
    raise ImportError('install yann[peft] to use LoRA') from error
  config = LoraConfig(
    r=rank,
    lora_alpha=alpha or rank * 2,
    lora_dropout=dropout,
    target_modules=targets,
    task_type=task_type,
    **kwargs,
  )
  return get_peft_model(model, config)


def save(model, path: str | Path) -> Path:
  path = Path(path)
  path.mkdir(parents=True, exist_ok=True)
  model.save_pretrained(path)
  return path


def load(model, path: str | Path, *, trainable: bool = True):
  try:
    from peft import PeftModel
  except ImportError as error:
    raise ImportError('install yann[peft] to load LoRA adapters') from error
  return PeftModel.from_pretrained(model, path, is_trainable=trainable)
