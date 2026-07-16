from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


def masked_sum(value: Tensor, mask: Tensor | None = None, dim=None, keepdim=False) -> Tensor:
  return value.sum(dim=dim, keepdim=keepdim) if mask is None else (value * mask).sum(dim=dim, keepdim=keepdim)


def masked_mean(value: Tensor, mask: Tensor | None = None, dim=None, keepdim=False) -> Tensor:
  if mask is None:
    return value.mean(dim=dim, keepdim=keepdim)
  denominator = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1)
  return masked_sum(value, mask, dim=dim, keepdim=keepdim) / denominator


def masked_var(value: Tensor, mask: Tensor | None = None, dim=None, unbiased=False) -> Tensor:
  mean = masked_mean(value, mask, dim=dim, keepdim=True)
  variance = masked_mean((value - mean).square(), mask, dim=dim)
  if unbiased and mask is not None:
    count = mask.sum(dim=dim).clamp_min(2)
    variance = variance * count / (count - 1)
  return variance


def whiten(value: Tensor, mask: Tensor | None = None, eps: float = 1e-8) -> Tensor:
  mean = masked_mean(value, mask)
  variance = masked_var(value, mask)
  output = (value - mean) * torch.rsqrt(variance + eps)
  return output if mask is None else output * mask


def token_logprobs(logits: Tensor, tokens: Tensor) -> Tensor:
  return F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def entropy(logits: Tensor, mask: Tensor | None = None) -> Tensor:
  log_probs = F.log_softmax(logits, dim=-1)
  probs = log_probs.exp()
  value = -(probs * log_probs).sum(dim=-1)
  return masked_mean(value, mask)


def kl_penalty(logprobs: Tensor, reference_logprobs: Tensor, mask: Tensor | None = None) -> Tensor:
  return masked_mean(logprobs - reference_logprobs, mask)


def discounted_returns(rewards: Tensor, gamma: float = 1.0, mask: Tensor | None = None) -> Tensor:
  returns = torch.zeros_like(rewards)
  running = torch.zeros_like(rewards[..., -1])
  for index in range(rewards.shape[-1] - 1, -1, -1):
    active = 1 if mask is None else mask[..., index]
    running = rewards[..., index] + gamma * running * active
    returns[..., index] = running
  return returns


def group_advantages(rewards: Tensor, groups: Tensor | Sequence[int] | None = None, eps: float = 1e-8) -> Tensor:
  if groups is None:
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(eps)
  groups = torch.as_tensor(groups, device=rewards.device)
  output = torch.empty_like(rewards)
  for group in groups.unique():
    select = groups == group
    values = rewards[select]
    output[select] = (values - values.mean()) / values.std(unbiased=False).clamp_min(eps)
  return output


def clipped_policy_loss(
  logprobs: Tensor,
  old_logprobs: Tensor,
  advantages: Tensor,
  *,
  clip_ratio: float = 0.2,
  mask: Tensor | None = None,
) -> Tensor:
  ratio = (logprobs - old_logprobs).exp()
  unclipped = ratio * advantages
  clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
  return -masked_mean(torch.minimum(unclipped, clipped), mask)


def correctness_gate(reward: Tensor, correct: Tensor, incorrect: float = -1.0) -> Tensor:
  return torch.where(correct.bool(), reward, torch.full_like(reward, incorrect))


def runtime_reward(
  baseline: Tensor,
  candidate: Tensor,
  *,
  max_speedup: float = 10.0,
  eps: float = 1e-8,
) -> Tensor:
  ratio = (baseline / candidate.clamp_min(eps)).clamp(max=max_speedup)
  return ratio.log()
