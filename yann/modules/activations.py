import torch
from torch import nn
from torch.nn import functional as F


def relu_squared_(x: torch.Tensor):
  return F.relu_(x).pow_(2)


def relu_squared(x: torch.Tensor):
  return F.relu(x).pow_(2)


class StarReLU(nn.Module):
  """
  Activation from https://arxiv.org/abs/2210.13452
  """

  def __init__(self):
    super().__init__()
    self.scale = nn.Parameter(torch.tensor(1))
    self.shift = nn.Parameter(torch.tensor(0))

  def forward(self, x: torch.Tensor):
    return self.scale * F.relu(x).square() + self.shift


def topk_softmax(logits: torch.Tensor, k, dim=-1, scatter=False):
  top = torch.topk(logits, k, largest=True, sorted=True, dim=dim)

  probs = F.softmax(top.values, dim=dim)

  if scatter:
    p = torch.zeros_like(logits)
    p.scatter_(dim, top.indices, probs)
    return p
  else:
    return probs, top.indices
