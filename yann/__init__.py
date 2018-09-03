__version__ = '0.0.4'

import torch
import numpy as np


def seed(val=1):
  import numpy as np
  import torch
  import random
  random.seed(val)
  np.random.seed(val)
  torch.manual_seed(val)
  try:
    torch.cuda.manual_seed(val)
  except: pass
  return val


def benchmark():
  from torch.backends import cudnn
  cudnn.benchmark = True


def detect_anomalies(val=True):
  import torch.autograd
  torch.autograd.set_detect_anomaly(val)


def resolve(x, modules=None, required=False, types=None,
            validate=None, **kwargs):
  if isinstance(x, str):
    for m in (modules or []):
      if hasattr(m, x):
        x = getattr(m, x)
        break

  if isinstance(x, type):
    x = x(**kwargs)

  if required:
    assert x, f'Got invalid argument, was required but got {str(x)}'

  if types:
    assert isinstance(x, types), f'Expected {types} for got {x} of type {type(x)}'

  if validate:
    assert validate(x), f'Failed validation, got {x}'

  return x


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


def trainable(parameters):
  return (p for p in parameters if p.requires_grad)


# TODO: handle batchnorm
def freeze(parameters):
  for p in parameters:
    p.requires_grad = False

def unfreeze(parameters):
  for p in parameters:
    p.requires_grad = True


def to_numpy(x):
  if isinstance(x, np.ndarray):
    return x
  if torch.is_tensor(x):
    return x.to('cpu').numpy()
  return np.array(x)



def to_fp16(model):
  # https://discuss.pytorch.org/t/training-with-half-precision/11815
  # https://github.com/csarofeen/examples/tree/fp16_examples_cuDNN-ATen/imagenet
  # https://github.com/NVIDIA/apex
  # https://github.com/NVIDIA/apex/tree/master/apex/amp
  model.half()
  for layer in model.modules():
    if isinstance(layer, torch.nn.BatchNorm2d):
      layer.float()


class lazy:
  __slots__ = 'method', 'name'

  def __init__(self, method):
    self.method = method
    self.name = method.__name__

  def __get__(self, obj, cls):
    if obj:
      val = self.method(obj)
      setattr(obj, self.name, val)
      return val