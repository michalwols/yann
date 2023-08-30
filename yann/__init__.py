import typing
from contextlib import contextmanager

__version__ = '0.0.40'

from typing import Union

import torch
from torch import nn
import numpy as np

from .config.defaults import default
from .config.setup import registry

register = registry
resolve = registry.resolve

from pathlib import Path

from yann.utils import to_numpy, repeat, counter, is_notebook, timeout
from yann.data import batches, shuffle, chunk
from yann.data.io import load, save
from yann.data.io.download import download
from yann.data.utils import pad, pad_to_largest
# from yann.data import datasets
from yann.viz import show, plot
from yann.utils.timer import time
from yann.utils.profile import profile, param_count

from yann.testing import Checker
from yann.data.loaders import loader

# T = torch.Tensor
#
# check = Checker()
#
#
#
# def serialize():
#   pass
#
# def deserialize():
#   pass
#
#
# class Cache:
#   def __call__(self, data):
#     pass
#
# def cache(data=None, path=None, recompute=None, validate=False, hash=True):
#   """
#
#   processed_data
#
#   yann.cache(processed_data
#
#
#   Args:
#     data:
#     path:
#     recompute:
#     validate:
#     hash:
#
#   Returns:
#
#   """
#
#
# def checkpoint():
#   pass
#
# def load_checkpoint():
#   pass


context = object()


memory_formats = dict(
  contiguous_format=torch.contiguous_format,
  channels_last=torch.channels_last,
  preserve_format=torch.preserve_format
)

def to_tensor(
    x: Union[list, tuple, np.ndarray, torch.Tensor, 'PIL.Image.Image']
) -> torch.Tensor:
  if torch.is_tensor(x):
    return x
  if isinstance(x, np.ndarray):
    return torch.from_numpy(x)
  import PIL.Image
  if isinstance(x, PIL.Image.Image):
    from torchvision.transforms import functional as F
    return F.to_tensor(x)
  return torch.Tensor(x)


def seed(val=1, deterministic=False):
  import numpy as np
  import torch
  import random
  random.seed(val)
  np.random.seed(val)
  torch.manual_seed(val)
  try:
    torch.cuda.manual_seed(val)

    if deterministic:
      torch.cuda.deterministic = True
      # torch.cuda.benchmark = False
  except:
    pass
  return val


def get_item(x: Union[torch.Tensor, np.ndarray]):
  if torch.is_tensor(x):
    if x.is_cuda:
      x = x.cpu()
    return x.item()
  elif isinstance(x, np.ndarray):
    return x.item()
  return x


def benchmark():
  from torch.backends import cudnn
  cudnn.benchmark = True


def detect_anomalies(val=True):
  import torch.autograd
  torch.autograd.set_detect_anomaly(val)


def evaluate(model, batches, device=None, transform=None):
  model = yann.resolve.model(model, required=True)
  if isinstance(batches, str):
    batches = yann.loader(batches, transform=transform)

  for x, y in batches:
    if device:
      x, y = x.to(device), y.to(device)

    model.eval()
    with torch.inference_mode():
      pred = model(x)

    yield x, y, pred


def predict_multicrop(model, inputs, reduce='mean'):
  batch_size, num_crops, *sample_shape = inputs.shape
  flat_preds = model(inputs.view(-1, *sample_shape))
  outputs = flat_preds.view(batch_size, num_crops, -1)
  if reduce:
    outputs = getattr(torch, reduce)(outputs, 1)
  return outputs


class Multicrop(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, inputs, *rest):
    return predict_multicrop(self.model, inputs)


def set_param(x, param, val):
  if hasattr(x, 'param_groups'):
    for group in x.param_groups:
      group[param] = val
  else:
    setattr(x, param, val)


def scale_param(x, param, mult):
  if hasattr(x, 'param_groups'):
    for group in x.param_groups:
      group[param] *= mult
  else:
    setattr(x, param, getattr(x, param) * mult)


def group_params(model, get_key):
  splits = {}
  for name, param in model.named_parameters():
    splits[get_key(name, param)] = param
  return splits

from torch.nn.modules.batchnorm import _BatchNorm


def split_regularization_params(
    module: nn.Module,
    excluded_modules=(_BatchNorm,),
    excluded_names=('bias',),
    param_groups=True,
    weight_decay=1e-4
):
  """
  filter out parameters which should not be regularized
  """
  reg, no_reg = [], []
  m: nn.Module
  for m in module.modules():
    if isinstance(m, excluded_modules):
      no_reg.extend((x for x in m.parameters(recurse=False) if x is not None))
    else:
      for name, param in m.named_parameters(recurse=False):
        if param is not None:
          if name.endswith(excluded_names):
            no_reg.append(param)
          else:
            reg.append(param)
  if param_groups:
    return [dict(params=reg, weight_decay=weight_decay), dict(params=no_reg, weight_decay=0)]
  else:
    return reg, no_reg


def trainable(parameters):
  return (p for p in parameters if p.requires_grad)


# TODO: handle batchnorm
def freeze(x, exclude=None):
  if isinstance(x, nn.Module):
    if exclude:
      for m in x.modules():
        if not isinstance(m, exclude):
          for p in m.parameters(recurse=False):
            p.requires_grad = False
    else:
      for p in x.parameters():
        p.requires_grad = False
  elif exclude:
    raise ValueError(
      "can't exclude modules if parameters are passed, "
      "pass an instance of nn.Module if you need to "
      "exclude certain modules")
  else:
    for p in x:
      p.requires_grad = False

  return x


def freeze_non_batchnorm(x):
  """
  Freeze layers except batchnorm
  should be used when fine tuning / transfer learning
  """
  return freeze(x, exclude=_BatchNorm)


def unfreeze(parameters):
  if isinstance(parameters, nn.Module):
    parameters = parameters.parameters()
  for p in parameters:
    p.requires_grad = True
  return parameters


def filter_modules(module: nn.Module, type, named=True):
  for n, m in module.named_modules():
    if isinstance(m, type):
      if named:
        yield n, m
      else:
        yield m


def replace_linear(model, num_outputs, layer_name=None):
  if layer_name is None:
    linear_layers = list(filter_modules(model, nn.Linear, named=True))
    if len(linear_layers) == 1:
      layer_name = linear_layers[0][0]
    elif len(linear_layers) == 0:
      raise ValueError('No linear layers found in model')
    else:
      raise ValueError(
        f'Multiple linear layers found and layer name was not provided, '
        f'provide a valid layer_name, '
        f'(valid names: {", ".join([n for n, m in linear_layers])})'
      )


  if '.' in layer_name:
    *path, layer_name = list(layer_name.split('.'))
    for p in path:
      model = getattr(model, p)

  old_linear = getattr(model, layer_name)
  new_linear = nn.Linear(old_linear.in_features, num_outputs)
  new_linear.to(old_linear.weight.device)

  setattr(
    model,
    layer_name,
    new_linear
  )

  return layer_name


def to_fp16(model):
  # https://discuss.pytorch.org/t/training-with-half-precision/11815
  # https://github.com/csarofeen/examples/tree/fp16_examples_cuDNN-ATen/imagenet
  # https://github.com/NVIDIA/apex
  # https://github.com/NVIDIA/apex/tree/master/apex/amp
  model.half()
  for layer in model.modules():
    if isinstance(layer, torch.nn.BatchNorm2d):
      layer.float()


def none_grad(model: nn.Module):
  """
  more efficient version of zero_grad()
  """
  for p in model.parameters():
    p.grad = None


@contextmanager
def eval_mode(*modules, grad=False):
  if grad:
    training = (m.training for m in modules)
    try:
      for m in modules: m.eval()
      yield
    finally:
      for m, train in zip(modules, training):
        if train:
          m.train()
  else:
    with torch.no_grad():
      training = (m.training for m in modules)
      try:
        for m in modules: m.eval()
        yield
      finally:
        for m, train in zip(modules, training):
          if train:
            m.train()


@contextmanager
def train_mode(*modules):
  initial_training_states = (m.training for m in modules)
  try:
    for m in modules:
      m.train()

    yield

  finally:
    for m, train in zip(modules, initial_training_states):
      if train:
        m.train()
      else:
        m.eval()


@contextmanager
def optim_step(optimizer, zero_grad=True):
  if zero_grad:
    optimizer.zero_grad()

  yield

  optimizer.step()


def to(items, **kwargs):
  """call `.to()` on all items that have a `to()` method, skips ones that don't"""
  if hasattr(items, 'to'):
    return items.to(**kwargs)
  elif isinstance(items, dict):
    return {k: to(v, **kwargs) for k, v in items.items()}
  elif isinstance(items, tuple):
    return tuple(to(x, **kwargs) for x in items)
  elif isinstance(items, list):
    return [to(x, **kwargs) for x in items]
  elif isinstance(items, np.ndarray):
    return to(torch.from_numpy(items), **kwargs)
  elif isinstance(items, (int, float)):
    return to(torch.as_tensor(items), **kwargs)
  return items


def get_device(module):
  return next(module.parameters()).device


def get_trainer(params=None, **kwargs):
  from yann.train import Trainer
  return Trainer(params=params, **kwargs)


def get_model_name(model):
  if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
    model = model.module

  if hasattr(model, 'name'):
    return model.name

  return model.__class__.__name__


def load_state_dict(
    x,
    state_dict: Union[str, 'pathlib.Path', dict],
    strict: bool = True
):
  if not isinstance(state_dict, dict):
    state_dict = yann.load(state_dict)

  return x.load_state_dict(state_dict, strict=strict)



def grad_norm(parameters, norm_type: float = 2.0):
  """

  Adapted from https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html
  Args:
    parameters:
    norm_type:

  Returns:

  """
  if isinstance(parameters, nn.Module):
    parameters = parameters.parameters()
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = [p for p in parameters if p.grad is not None]
  parameters = list(parameters)

  norm_type = float(norm_type)
  if len(parameters) == 0:
    return torch.tensor(0.)
  device = parameters[0].grad.device
  try:
    from torch import inf
  except ImportError:
    from torch._six import inf

  if norm_type == inf:
    norms = [p.grad.detach().abs().max().to(device) for p in parameters]
    norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
  else:
    norm = torch.norm(
      torch.stack([
        torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters
      ]),
      norm_type
    )
  return norm


def param_norm(parameters, norm_type: float = 2.0):
  if isinstance(parameters, nn.Module):
    parameters = parameters.parameters()
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]

  parameters = list(parameters)
  norm_type = float(norm_type)
  if len(parameters) == 0:
    return torch.tensor(0.)
  device = parameters[0].device
  try:
    from torch import inf
  except ImportError:
    from torch._six import inf
  if norm_type == inf:
    norms = [p.detach().abs().max().to(device) for p in parameters]
    norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
  else:
    norm = torch.norm(
      torch.stack([
        torch.norm(p.detach(), norm_type).to(device) for p in parameters
      ]),
      norm_type
    )
  return norm


def nested_lookup(obj, key):
  keys = key.split('.')

  for k in keys:
    if isinstance(obj, typing.Mapping):
      obj = obj[k]
    elif isinstance(obj, typing.Sequence):
      obj = obj[int(k)]
    elif hasattr(obj, k):
      obj = getattr(obj, k)
    else:
      raise KeyError(f'{key} not found')
  return obj


import yann.params
import yann.metrics
import yann.train
import yann.callbacks
import yann.optim