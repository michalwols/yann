import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch import nn

from ..data.classes import smooth as label_smoothing


def soft_target_cross_entropy(
    inputs,
    targets,
    smooth=None,
    reduce=True,
    dim=1,
    reduction='elementwise_mean'):
  """"like cross_entropy but using soft targets"""
  if smooth:
    targets = label_smoothing(targets, smooth)

  vals = torch.sum(-targets * F.log_softmax(inputs, dim=dim), dim=dim)
  if not reduce:
    return vals

  if reduction == 'mean':
    return torch.sum(vals)
  elif reduction == 'elementwise_mean':
    return torch.mean(vals)
  elif 'none':
    return vals
  raise ValueError('Unsupported reduction mode: ' + str(reduction))


class SoftTargetCrossEntropyLoss(_Loss):
  def __init__(self, smooth=None, reduce=True, dim=1, reduction='elementwise_mean'):
    super().__init__(reduce=reduce, reduction=reduction)
    self.reduce = reduce
    self.reduction = reduction
    self.smooth = smooth
    self.dim = dim

  def forward(self, inputs, targets):
    return soft_target_cross_entropy(
      inputs,
      targets,
      smooth=self.smooth,
      reduce=self.reduce,
      dim=self.dim,
      reduction=self.reduction)



def contrastive_loss():
  pass


def triplet_loss():
  pass