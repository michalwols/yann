from __future__ import annotations

from typing import Any

import torch


def _dtype(value: str | torch.dtype | None):
  if isinstance(value, torch.dtype) or value is None:
    return value
  return {
    'fp32': torch.float32,
    'float32': torch.float32,
    'fp16': torch.float16,
    'float16': torch.float16,
    'bf16': torch.bfloat16,
    'bfloat16': torch.bfloat16,
  }[value]


def load(
  name: str,
  *,
  task: str = 'causal-lm',
  tokenizer: bool = True,
  dtype: str | torch.dtype | None = None,
  **kwargs: Any,
):
  try:
    from transformers import (
      AutoModel,
      AutoModelForCausalLM,
      AutoModelForSequenceClassification,
      AutoTokenizer,
    )
  except ImportError as error:
    raise ImportError(
      'install yann[transformers] to load Hugging Face models'
    ) from error

  model_cls = {
    'model': AutoModel,
    'causal-lm': AutoModelForCausalLM,
    'sequence-classification': AutoModelForSequenceClassification,
  }.get(task)
  if model_cls is None:
    raise ValueError(f'unsupported task {task!r}')
  if dtype is not None:
    kwargs['torch_dtype'] = _dtype(dtype)
  model = model_cls.from_pretrained(name, **kwargs)
  if not tokenizer:
    return model
  return model, AutoTokenizer.from_pretrained(name)
