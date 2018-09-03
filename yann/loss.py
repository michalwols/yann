import torch.nn.functional as F
import torch
from torch.nn.modules.loss import _Loss


def soft_target_cross_entropy(input, target, reduce=True,
                              reduction='elementwise_mean'):
  """"like cross_entropy but using soft tarets"""
  vals = torch.sum(-target * F.log_softmax(input, dim=1))
  if not reduce:
    return vals

  if reduction == 'mean':
    return torch.sum(vals)
  elif reduction == 'elementwise_mean':
    return torch.mean(vals)
  elif 'none':
    return vals
  raise ValueError('Unsupported reduction mode: ' + str(reduction))



class SoftTargetCrossEntopyLoss(_Loss):
  def __init__(self, reduce=None, reduction='elementwise_mean'):
    super().__init__(reduce=reduce, reduction=reduction)
    self.reduce = reduce
    self.reduction = reduction

  def forward(self, input, target):
    return soft_target_cross_entropy(input, target, reduce=self.reduce,
                                     reduction=self.reduction)
