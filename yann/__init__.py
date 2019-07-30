from contextlib import contextmanager

__version__ = '0.0.25'

import torch
from torch import nn
from .config.setup import registry

register = registry
resolve = registry.resolve

from .utils import to_numpy

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
  except:
    pass
  return val


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
  for group in x.param_groups:
    group[param] = val


def scale_param(x, param, mult):
  for group in x.param_groups:
    group[param] *= mult


def trainable(parameters):
  return (p for p in parameters if p.requires_grad)


# TODO: handle batchnorm
def freeze(parameters):
  for p in parameters:
    p.requires_grad = False


def unfreeze(parameters):
  for p in parameters:
    p.requires_grad = True


def replace_linear(model, num_outputs, layer_name='last_linear'):
  setattr(
    model,
    layer_name,
    nn.Linear(
      getattr(model, layer_name).in_features,
      num_outputs)
  )


def to_fp16(model):
  # https://discuss.pytorch.org/t/training-with-half-precision/11815
  # https://github.com/csarofeen/examples/tree/fp16_examples_cuDNN-ATen/imagenet
  # https://github.com/NVIDIA/apex
  # https://github.com/NVIDIA/apex/tree/master/apex/amp
  model.half()
  for layer in model.modules():
    if isinstance(layer, torch.nn.BatchNorm2d):
      layer.float()


class HyperParams:
  def __init__(self, **args):
    self.__dict__.update(args)

  def __setattr__(self, key, value):
    raise AttributeError('Updating properties is not permitted')

  def __getitem__(self, item):
    if isinstance(item, (tuple, list)):
      return tuple(self.__dict__[k] for k in item)
    return self.__dict__[item]

  def fork(self, **args):
    return HyperParams(**{**self.__dict__, **args})

  def __repr__(self):
    return (
      'HyperParams('
      f"{', '.join(f'{k}={v}' for k, v in self.__dict__.items())}"
      ')'
    )

  def __str__(self):
    return (
        'HyperParams (\n' +
        ''.join('  {}={}\n'.format(k, v) for k, v in self.__dict__.items()) +
        ')'
    )

  def __len__(self):
    return len(self.__dict__)

  def __hash__(self):
    return hash(tuple(sorted(self.items())))

  def keys(self):
    return self.__dict__.keys()

  def values(self):
    return self.__dict__.values()

  def items(self):
    return self.__dict__.items()

  def grid(self, **args):
    raise NotImplementedError()

  def inject(self, scope=None, uppercase=True):
    scope = globals() if scope is None else scope
    for k, v in self.items():
      scope[k.upper() if uppercase else k] = v

  @classmethod
  def collect(cls, scope=None, types=(int, str, float, bool),
              upper_only=True, lowercase=True):
    scope = globals() if scope is None else scope

    d = {}
    for k, v in scope.items():
      if types and not isinstance(v, types):
        continue
      if upper_only and not k.isupper():
        continue

      d[k.lower() if lowercase else k] = v

    return cls(**d)

  def __eq__(self, other):
    return hash(self) == hash(other)


@contextmanager
def evalmode(*modules, grad=False):
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
