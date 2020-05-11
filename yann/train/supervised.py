import logging

import datetime
import pathlib
import uuid

import torch
import torch.nn
from torch.utils.data import DataLoader
from typing import Optional
import types
from ..callbacks import get_callbacks
from .. import callbacks as yann_callbacks
from .. import resolve, evaluate, to, default, trainable
from ..data import get_dataset_name, Classes
from ..datasets import TransformDataset
from ..data.io import save_json
from ..export import export
from ..inference.predict import Classifier
from ..train.base import BaseTrainer
from ..utils.decorators import lazy
from ..utils.ids import memorable_id
from ..utils import counter, print_tree, timestr, hash_params


def get_model_name(model):
  if isinstance(model, torch.nn.DataParallel):
    model = model.module

  if hasattr(model, 'name'):
    return model.name

  return model.__class__.__name__


class Paths:
  # TODO: add a way to display the directory tree
  def __init__(self, root):
    self.root = pathlib.Path(root)

    self.checkpoint_format = '{}{}'

  def create(self):
    self.root.mkdir(parents=True, exist_ok=True)

  @property
  def checkpoints(self):
    path = self.root / 'checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    return path

  def checkpoint(self, **kwargs):
    return self.checkpoints / self.checkpoint_format.format(**kwargs)

  @property
  def tensorboard(self):
    return self.root / 'tensorboard'

  @property
  def logs(self):
    return self.root / 'logs'

  @property
  def evals(self):
    return self.root / 'evals'

  @property
  def plots(self):
    return self.root / 'plots'

  @property
  def outputs(self):
    return self.root / 'outputs'

  @property
  def exports(self):
    return self.root / 'exports'

  @property
  def summary(self):
    return self.root / 'summary.json'

  def tree(self, **kwargs):
    print_tree(self.root, **kwargs)


class Trainer(BaseTrainer):
  model: torch.nn.Module
  lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
  loader: DataLoader

  def __init__(
      self,
      model,
      dataset=None,
      optimizer=None,
      loss=None,
      loader=None,
      sampler=None,
      num_workers=8,
      transform=None,
      transform_batch=None,
      lr_scheduler=None,
      lr_batch_step=False,
      callbacks=None,
      device=None,
      parameters='trainable',
      batch_size=16,
      val_dataset=None,
      val_loader=None,
      val_transform=None,
      classes=None,
      parallel=False,
      name=None,
      description=None,
      root='./experiments/train-runs/',
      metrics=None,
      collate=None,
      params=None,
      pin_memory=True,
      step=None,
      id=None
  ):
    super().__init__()

    self.id = id or memorable_id()

    self.params = params

    self.model = resolve.model(
      model,
      required=True,
      validate=callable
    )
    if parallel:
      self.model = torch.nn.DataParallel(self.model)

    self.loss = resolve.loss(
      loss,
      required=True,
      validate=callable,
    )

    if parameters == 'trainable' and self.model:
      parameters = trainable(self.model.parameters())
    self.optimizer = resolve.optimizer(
      optimizer,
      args=(parameters or self.model.parameters(),),
      required=True,
      validate=lambda x: hasattr(x, 'step')
    )

    self.dataset = resolve.dataset(
      dataset,
      required=not loader,
    )

    if classes:
      self.classes = (
        classes if isinstance(classes, Classes)
        else Classes(classes)
      )
    elif hasattr(self.dataset, 'classes') and isinstance(self.dataset.classes, Classes):
      self.classes = self.dataset.classes
    else:
      self.classes = None

    if transform:
      self.dataset = TransformDataset(self.dataset, transform)
    self.transform = transform

    self.transform_batch = transform_batch

    self.batch_size = batch_size

    self.loader = loader or DataLoader(
      self.dataset,
      batch_size=self.batch_size,
      pin_memory=pin_memory,
      shuffle=False if sampler else True,
      sampler=sampler,
      num_workers=num_workers,
      **({'collate_fn': collate} if collate else {})
    )

    self.val_dataset = resolve.dataset(
      val_dataset,
      required=False,
    )

    self.val_transform = val_transform or transform
    if self.val_transform:
      self.val_dataset = TransformDataset(self.val_dataset, self.val_transform)

    self.val_loader = val_loader or (val_dataset and DataLoader(
      self.val_dataset,
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=num_workers
    ))

    self.lr_scheduler = resolve.lr_scheduler(
      lr_scheduler,
      kwargs=dict(optimizer=self.optimizer)
    )
    self.lr_batch_step = lr_batch_step

    if step is not None:
      self.override(self.step, step)

    self.num_samples = 0
    self.num_steps = 0
    self.num_epochs = 0

    self.time_created = datetime.datetime.utcnow()

    self.name = name or (
      f"{get_dataset_name(self.loader)}-{get_model_name(self.model)}"
    )
    self.description = description

    self.paths = Paths(pathlib.Path(root)  / self.name / timestr(self.time_created))
    self.paths.create()

    self.history = None
    has_history = False
    if callbacks:
      if callbacks is True:
        # if callbacks=True use the default set of callbacks
        callbacks = get_callbacks()
      for c in callbacks:
        if isinstance(c, yann_callbacks.History):
          self.history = c
          has_history = True
          break
    metrics = (metrics,) if isinstance(metrics, str) else metrics
    self.history = self.history or yann_callbacks.History(*(metrics or ()))

    self.callbacks = callbacks or [
      yann_callbacks.Logger()
    ]

    if not has_history:
      self.callbacks.append(self.history)

    # make sure history callback is called first
    self.callbacks = sorted(
      self.callbacks,
      key=lambda x: 0 if x is self.history else 1)

    self.function_callback = None

    device = default.device if device is None else device
    if device:
      self.device = torch.device(device) if isinstance(device, str) else device
      self.to(self.device)

  @property
  def root(self):
    """for backwards compatibility, self.paths.root used to be on self.root"""
    return self.paths.root

  def __setattr__(self, key, value):
    if key == 'optimizer':
      if hasattr(self, 'lr_scheduler') and hasattr(self.lr_scheduler, 'optimizer'):
        self.lr_scheduler.optimizer = value
    if key == 'loader':
      if hasattr(self, 'dataset') and hasattr(value, 'dataset'):
        super(Trainer, self).__setattr__('dataset', value.dataset)
      if hasattr(value, 'batch_size'):
        super(Trainer, self).__setattr__('batch_size', value.batch_size)
    if key == 'batch_size' and hasattr(self, 'batch_size') and self.batch_size != key:
      if hasattr(self, 'loader') and self.loader:
        raise ValueError(
          'Cannot modify batch_size because a loader is defined '
          'and modifying batch size of a loader is not supported, '
          'try creating and setting a new loader instead'
        )
      if key == 'dataset' and hasattr(self, 'dataset') and self.dataset \
          and hasattr(self, 'loader') and self.loader:
        raise ValueError(
          'Cannot modify dataset because a loader is defined '
          'and modifying dataset of a loader is not supported, '
          'try creating and setting a new loader instead'
        )

    logging.debug(f"setting '{key}' to {value}")
    super(Trainer, self).__setattr__(key, value)

  # @classmethod
  # def from_params(cls, params):
  #   pass

  def to(self, device=None):
    self.device = device

    to(
      self.model,
      self.loss,
      self.optimizer,
      device=self.device
    )

  def on(self, event, callback=None):
    if not self.function_callback:
      self.function_callback = yann_callbacks.FunctionCallback()
      self.callbacks.append(self.function_callback)

    if callback:
      self.function_callback.on(event, callback)
      return self
    else:
      def decorated(func):
        self.function_callback.on(event, func)
        return func

      return decorated

  def train_mode(self):
    self.model.train()

  def eval_mode(self):
    self.model.eval()

  def epochs(self, num=None):
    for e in counter(start=self.num_epochs, end=self.num_epochs + num):
      yield e
      self.num_epochs += 1

  def batches(self, device=None):
    device = device or self.device

    for batch in self.loader:
      if self.transform_batch:
        batch = self.transform_batch(*batch)

      if device:
        yield to(*batch, device=device)
      else:
        yield batch

      self.num_steps += 1
      self.num_samples += len(batch[0])

  def override(self, method, function=False):
    """
    Override a method of the trainer
    Args:
      method: str or method reference
      function: function to be used as a replacement for the given method
    """
    method = method if isinstance(method, str) else method.__name__
    if not hasattr(self, method):
      raise AttributeError(f"Can't override method '{method}' because it's not defined")
    if function is False:
      # assume it's used as a decorator
      # @train.override('step')
      # def custom_step(trainer, inputs, targets):
      def decorator(f):
        setattr(self, method, types.MethodType(f, self))
      return decorator
    else:
      setattr(self, method, types.MethodType(function, self))

  def step(self, inputs, target):
    if not self.model.training:
      self.model.train()

    self.optimizer.zero_grad()

    outputs = self.model(inputs)
    loss = self.loss(outputs, target)
    loss.backward()
    self.optimizer.step()

    return outputs, loss

  def validate(self, loader=None, device=None):
    loader = loader or self.val_loader
    device = device or self.device

    self.model.eval()

    self.on_validation_start()

    ts, os = [], []
    with torch.no_grad():
      for inputs, targets, outputs in evaluate(self.model, loader, device):
        self.on_validation_batch(inputs=inputs, targets=targets,
                                 outputs=outputs)
        ts.append(targets)
        os.append(outputs)

      ts = torch.cat(ts)
      os = torch.cat(os)
      loss = self.loss(os, ts)
    self.on_validation_end(targets=ts, outputs=os, loss=loss)
    return loss

  def run(self, epochs=None):
    self._stop = False
    try:
      self.on_train_start()

      for _ in self.epochs(num=epochs):
        self.on_epoch_start(epoch=self.num_epochs)
        for batch_idx, (inputs, targets) in enumerate(self.batches()):
          self.on_batch_start(
            batch=batch_idx,
            inputs=inputs,
            targets=targets
          )
          try:
            outputs, loss = self.step(inputs, targets)
          except KeyboardInterrupt as e:
            self.stop()
            break
          except Exception as e:
            self.on_batch_error(batch=batch_idx, error=e)
            if self._stop: break
            raise e

          self.on_batch_end(
            batch=batch_idx,
            inputs=inputs,
            targets=targets,
            outputs=outputs,
            loss=loss
          )

          if self.lr_scheduler and self.lr_batch_step:
            self.lr_scheduler.step(epoch=self.num_steps)

          if self._stop: break
        if self._stop: break

        if self.val_loader:
          val_loss = self.validate()
          if self.lr_scheduler and not self.lr_batch_step:
            if 'metrics' in self.lr_scheduler.step.__code__.co_varnames:
             self.lr_scheduler.step(metrics=val_loss, epoch=self.num_epochs)
            else:
              self.lr_scheduler.step(epoch=self.num_epochs)
        elif self.lr_scheduler and not self.lr_batch_step:
          self.lr_scheduler.step(epoch=self.num_epochs)

        self.on_epoch_end(epoch=self.num_epochs)

      self.on_train_end()
    except Exception as e:
      self.on_error(e)
      raise e

  def __call__(self, *args, **kwargs):
    self.run(*args, **kwargs)

  def stop(self):
    self._stop = True

  @lazy
  def predict(self):
    return Classifier(
      model=self.model,
      classes=self.classes,
      preprocess=self.transform,
      postprocess=None
    )

  def checkpoint(self, name=None):
    state = self.state_dict()
    path = str(self.paths.checkpoints / (
      f"{name}.th" if name else
      f"{timestr()}-epoch-{self.num_epochs:03d}-steps-{self.num_steps:05d}.th"
    ))
    torch.save(state, path)
    return path

  def load_checkpoint(self, path, metadata=True):
    # TODO: add 'latest', 'best' support
    data = torch.load(path)
    self.load_state_dict(data, metadata=metadata)

  def export(self, path=None, trace=False, meta=None, postprocess=None):
    path = path or self.paths.exports / timestr()
    export(
      model=self.model,
      preprocess=self.val_transform,
      postprocess=postprocess,
      classes=self.classes,
      trace=trace and next(iter(self.loader)),
      path=path,
      meta=dict(
        name=self.name,
        root=str(self.paths.root),
        dataset=get_dataset_name(self.dataset),
        num_steps=self.num_steps,
        **(meta or {})
      )
    )

    return path

  def state_dict(self):
    data = {
      'metadata': {
        'id': self.id,
        'num_steps': self.num_steps,
        'num_samples': self.num_samples,
        'num_epochs': self.num_epochs,
        'batch_size': self.batch_size,
        'name': self.name,
        'time_created': self.time_created,
        'param_hash': hash_params(self.model)
      }
    }

    for k, v in self.__dict__.items():
      if hasattr(v, 'state_dict'):
        data[k] = {
          'state_dict': v.state_dict(),
          'class': v.__class__.__name__
        }

    return data

  def load_state_dict(self, data, metadata=True):
    """
    TODO: add a way to specify which parts should be loaded (ex: model only)
    """
    skipped = set()
    for k, v in data.items():
      if 'state_dict' in v and hasattr(self, k):
        getattr(self, k).load_state_dict(v['state_dict'])
      else:
        skipped.add(k)

    if metadata and 'metadata' in data:
      skipped.discard('metadata')
      for k, v in data['metadata'].items():
        setattr(self, k, v)

    if skipped:
      logging.warning(f'skipped {skipped} when loading checkpoint')

  def summary(self):
    summary = {
      'id': self.id,
      'name': self.name,
      'root': str(self.root),
      'num_steps': self.num_steps,
      'num_samples': self.num_samples,
      'num_epochs': self.num_epochs,
      'time_created': str(self.time_created),
      'params': dict(self.params) if self.params else {},
      'metrics': {
        'train': {
          k: ({'min': min(v), 'max': max(v)} if len(v) else {}) for k, v in self.history.metrics.items()
        },
        'validation': {
          k: ({'min': min(v), 'max': max(v)} if len(v) else {}) for k, v in self.history.val_metrics.items()
        }
      }
    }

    summary['duration'] = None
    if len(self.history.metrics.times) > 1:
      try:
        summary['duration'] = self.history.metrics.times[-1] - self.history.metrics.times[0]
      except:
        pass

    return summary

  def save_summary(self):
    save_json(self.summary(), self.paths.summary)
    return self.paths.summary

  def __str__(self):
    return f"""
id: {self.id}
name: {self.name}
root: {self.root}
batch_size: {self.batch_size}
device: {self.device}

MODEL
=====

{self.model}


DATASET
=======

{self.loader.dataset}


LOADER
======

{self.loader}

LOSS
====

{self.loss}


OPTIMIZER
=========

{self.optimizer}

SCHEDULER
=========

{self.lr_scheduler}


PROGRESS
========
epochs: {self.num_epochs}
steps: {self.num_steps}
samples: {self.num_samples}
"""

  def __repr__(self):
    return (
      f"Trainer("
      f"\n  name={self.name},"
      f"\n  root={self.root},"
      f"\n  batch_size={self.batch_size},"
      f"\n  device={self.device}"
      "\n)"
    )