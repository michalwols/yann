from contextlib import contextmanager

__version__ = '0.0.36'

import torch
from torch import nn
from .config.setup import registry

register = registry
resolve = registry.resolve

from .utils import to_numpy, repeat, counter
import numpy as np

from .data import batches, shuffle, chunk
from .data.io import load, save
from .data.utils import pad, pad_to_largest
from .data import datasets
from .viz import show, plot


from .testing import Checker

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


default_device = torch.device('cuda') \
  if torch.cuda.is_available() else torch.device('cpu')


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


def get_item(x):
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


def evaluate(model, batches, device=None):
  for x, y in batches:
    if device:
      x, y = x.to(device), y.to(device)

    model.eval()
    with torch.no_grad():
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
    return [dict(params=reg), dict(params=no_reg, weight_decay=0)]
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
  if not hasattr(model, layer_name):
    raise ValueError(f'Model does not have a linear layer named "{layer_name}"')
  setattr(
    model,
    layer_name,
    nn.Linear(
      getattr(model, layer_name).in_features,
      num_outputs)
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


def to(*items, **kwargs):
  """call `.to()` on all items that have a `to()` method, skips ones that don't"""
  return tuple(x.to(**kwargs) if hasattr(x, 'to') else x for x in items)
