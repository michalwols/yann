from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .rl import masked_mean


def causal_lm_loss(
  logits: Tensor,
  labels: Tensor,
  *,
  mask: Tensor | None = None,
  ignore_index: int = -100,
) -> Tensor:
  token_loss = F.cross_entropy(
    logits.reshape(-1, logits.shape[-1]),
    labels.reshape(-1),
    ignore_index=ignore_index,
    reduction='none',
  ).reshape(labels.shape)
  if mask is None:
    mask = labels.ne(ignore_index)
  return masked_mean(token_loss, mask)


def masked_kl(
  student_logits: Tensor,
  teacher_logits: Tensor,
  mask: Tensor | None = None,
  *,
  temperature: float = 1.0,
) -> Tensor:
  student_logprobs = F.log_softmax(student_logits / temperature, dim=-1)
  teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
  token_loss = F.kl_div(
    student_logprobs,
    teacher_probs,
    reduction='none',
  ).sum(dim=-1)
  return masked_mean(token_loss, mask) * temperature**2


def dpo_loss(
  chosen_logprobs: Tensor,
  rejected_logprobs: Tensor,
  reference_chosen_logprobs: Tensor,
  reference_rejected_logprobs: Tensor,
  *,
  beta: float = 0.1,
  label_smoothing: float = 0.0,
) -> Tensor:
  policy_ratio = chosen_logprobs - rejected_logprobs
  reference_ratio = reference_chosen_logprobs - reference_rejected_logprobs
  logits = beta * (policy_ratio - reference_ratio)
  positive = -F.logsigmoid(logits)
  negative = -F.logsigmoid(-logits)
  return ((1 - label_smoothing) * positive + label_smoothing * negative).mean()


def policy_delta(
  before_logits: Tensor,
  after_logits: Tensor,
  *,
  clip: float | None = None,
) -> Tensor:
  delta = F.log_softmax(after_logits, dim=-1) - F.log_softmax(before_logits, dim=-1)
  return delta.clamp(-clip, clip) if clip is not None else delta


def direct_opd_loss(
  student_logits: Tensor,
  anchor_logits: Tensor,
  delta: Tensor,
  mask: Tensor | None = None,
  *,
  scale: float = 1.0,
  temperature: float = 1.0,
  clip: float | None = None,
) -> Tensor:
  if clip is not None:
    delta = delta.clamp(-clip, clip)
  target_logits = anchor_logits / temperature + scale * delta
  target_probs = F.softmax(target_logits, dim=-1)
  student_logprobs = F.log_softmax(student_logits / temperature, dim=-1)
  token_loss = -(target_probs * student_logprobs).sum(dim=-1)
  return masked_mean(token_loss, mask) * temperature**2


__all__ = [
  'causal_lm_loss',
  'masked_kl',
  'dpo_loss',
  'policy_delta',
  'direct_opd_loss',
]
