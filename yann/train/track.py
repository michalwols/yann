from typing import Mapping, Any

import yann


class Tracker:
  """
  Simple callable that takes a trainer instance as input and returns a dict of values to log
  """
  freq: int = None

  def __call__(self, trainer: 'yann.train.Trainer') -> Mapping[str, Any]:
    raise NotImplementedError


class OptimizerState(Tracker):
  def __init__(
      self,
      optimizer=None,
      prefix='',
      keys=('lr', 'weight_decay', 'momentum', 'betas', 'alpha')):
    self.optimizer = optimizer
    self.keys = keys
    self.prefix = prefix

  def __call__(self, trainer: 'import yann.train.supervised.Trainer'):
    optim = self.optimizer or trainer.optimizer
    if not optim: return {}

    values = {}

    for n, group in enumerate(optim.param_groups):
      for key in self.keys:
        if key in group:
          value = group[key]
          if isinstance(value, (tuple, list)):
            for m, v in enumerate(value):
              values[f'{self.prefix}optimizer.param_groups.{n}.{key}.{m}'] = v
          else:
            values[f'{self.prefix}optimizer.param_groups.{n}.{key}'] = value

    return values


class ParamNorms(Tracker):
  def __init__(self, model=None, key='param_norm'):
    self.model = model
    self.key = key

  def __call__(self, trainer):
    import yann
    model = self.model or trainer.model

    return {self.key: yann.param_norm(model)}


class GradNorms(Tracker):
  def __init__(self, model=None, key='grad_norm'):
    self.model = model
    self.key = key

  def __call__(self, trainer):
    import yann
    model = self.model or trainer.model

    return {self.key: yann.grad_norm(model)}


class Keys(Tracker):
  def __init__(self, keys):
    self.keys = keys

  def __call__(self, trainer):
    values = {}
    for k in self.keys:
      try:
        values[k] = yann.nested_lookup(trainer, k)
      except:
        pass
    return values

