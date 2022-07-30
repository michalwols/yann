"""
Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py
"""
import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


def unitwise_norm(x: torch.Tensor, p=2.0):
  if x.ndim <= 1:
    return x.norm(p)
  else:
    return x.norm(p, dim=tuple(range(1, x.ndim)), keepdim=True)


def clip_grad_adaptive_(parameters, value=.01, norm_type=2.0, eps=1e-3):
  """
  Adaptive grad clipping
  """
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = [p for p in parameters if p.grad is not None]
  if len(parameters) == 0:
    return torch.tensor(0.)
  for param in parameters:
    weights, grads = param.detach(), param.grad.detach()
    max_norm = (
      unitwise_norm(weights, p=norm_type)
        .clamp_(min=eps)
        .mul_(value)
    )
    grad_norm = unitwise_norm(grads, p=norm_type)
    clipped_grad = grads * (max_norm / grad_norm.clamp(min=1e-6))
    new_grads = torch.where(grad_norm < max_norm, grads, clipped_grad)
    param.grad.detach().copy_(new_grads)


def clip_grad_(parameters, value, norm_type=2.0, mode='adaptive'):
  if mode == 'adaptive':
    return clip_grad_adaptive_(
      parameters=parameters,
      value=value,
      norm_type=norm_type
    )
  elif mode == 'norm':
    return clip_grad_norm_(
      parameters=parameters,
      max_norm=value,
      norm_type=norm_type
    )
  elif mode == 'value':
    return clip_grad_value_(
      parameters=parameters,
      clip_value=value,
    )
  else:
    raise ValueError(f'Unsupported mode={mode}, must be adaptive, norm or value')


class GradClipper:
  def __init__(self, value, norm_type=2.0, mode='adaptive'):
    """

    Args:
      value:
      norm_type:
      mode: 'adaptive' | 'norm' | 'value'
    """
    self.value = value
    self.norm_type = norm_type
    self.mode = mode

  def __call__(self, parameters):
    return clip_grad_(
      parameters=parameters,
      value=self.value,
      norm_type=self.norm_type,
      mode=self.mode
    )

  def state_dict(self):
    return self.__dict__

  def load_state_dict(self, dict):
    self.__dict__.update(dict)