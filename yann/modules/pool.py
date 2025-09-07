import torch
from torch import nn
from torch.nn import functional as F


def mac(batch):
  """MAC Pooling"""
  return F.adaptive_max_pool2d(batch, (1, 1))


def spoc(batch):
  """SPoC Pooling"""
  return F.adaptive_avg_pool2d(batch, (1, 1))


def generalized_mean(batch, p=3, eps=1e-8):
  """Generalized Mean Pooling (GeM)
  p=1 === spoc
  larger p ==> mac

  larger p leads to more localized (max) features
  """
  return F.adaptive_avg_pool2d(batch.clamp(min=eps) ** p, (1, 1)) ** (1 / p)


gem = generalized_mean


class GeM(nn.Module):
  def __init__(self, p=3, eps=1e-8):
    super(GeM, self).__init__()
    self.p = nn.Parameter(torch.Tensor([p]))
    self.eps = eps

  def forward(self, x):
    return gem(x, p=self.p, eps=self.eps)
