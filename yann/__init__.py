
import torch
import numpy as np

def seed(val=1):
  import numpy as np
  import torch
  torch.manual_seed(val)
  np.random.seed(val)
  return val


def benchmark():
  from torch.backends import cudnn
  cudnn.benchmark = True


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


def step(model, x, y, optimizer, criterion):
  model.train()
  optimizer.zero_grad()

  pred = model(x)
  loss = criterion(pred, y)

  loss.backward()
  optimizer.step()

  return x, y, pred, loss


def train(model, batches, optimizer, criterion, device=None):
  for x, y in batches:
    if device:
      x, y = x.to(device), y.to(device)
    yield step(model, x, y, optimizer, criterion)


def evaluate(model, batches, device=None):
  for x, y in batches:
    if device:
      x, y = x.to(device), y.to(device)

    model.eval()
    with torch.no_grad():
      pred = model(x)

    yield x, y, pred


def set_param(x, param, val):
  for group in x.param_groups:
    group[param] = val


def trainable(parameters):
  return (p for p in parameters if p.requires_grad)


def freeze(parameters):
  for p in parameters:
    p.requires_grad = False


def to_numpy(x):
  if isinstance(x, np.ndarray):
    return x
  if torch.is_tensor(x):
    return x.to('cpu').numpy()
  return np.array(x)