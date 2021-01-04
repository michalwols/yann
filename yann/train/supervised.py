import datetime
import logging
import pathlib
import torch
import torch.nn
import types
from statistics import mean
from torch.optim.optimizer import Optimizer
from torch.utils.data import Sampler
from typing import Optional, Callable

from yann.utils import fully_qualified_name
from yann.data.loaders import DataLoader
from .. import callbacks as yann_callbacks
from .. import resolve, evaluate, to, default, trainable
from ..callbacks import get_callbacks
from ..data import get_dataset_name, Classes
from ..data.io import save_json
from ..datasets import TransformDataset
from ..export import export
from ..params import HyperParams
from ..train.base import BaseTrainer
from ..utils import counter, print_tree, timestr, hash_params
from ..utils.ids import memorable_id


def get_model_name(model):
  if isinstance(model, torch.nn.DataParallel):
    model = model.module

  if hasattr(model, 'name'):
    return model.name

  return model.__class__.__name__


class Events:
  train_start = 'train_start'
  train_end = 'train_end'
  epoch_start = 'epoch_start'
  epoch_end = 'epoch_end'
  batch_start = 'batch_start'
  batch_end = 'batch_end'
  batch_error = 'batch_error'
  error = 'error'


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
  loss: Callable
  optimizer: Optional[Optimizer]

  lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
  loader: DataLoader
  classes: Optional[Classes]
  sampler: Optional[Sampler]
  params: Optional[HyperParams]
  paths: Paths


  def __init__(
      self,
      model=None,
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
    """

    Args:
      model: model to train
      dataset:
      optimizer:
      loss:
      loader:
      sampler:
      num_workers:
      transform:
      transform_batch: transform function that augments a batch of data
      lr_scheduler:
      lr_batch_step: if true will call optimizer.step() after each batch,
        otherwise will call it at the end of each epoch
      callbacks:
      device:
      parameters:
      batch_size:
      val_dataset:
      val_loader:
      val_transform:
      classes:
      parallel:
      name:
      description:
      root:
      metrics:
      collate:
      params:
      pin_memory:
      step:
      id:
    """
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

    self._use_callbacks = True

    device = default.device if device is None else device
    if device:
      self.device = torch.device(device) if isinstance(device, str) else device
      self.to(self.device)

  @classmethod
  def from_params(cls, params, **kwargs):
    return cls(**params, **kwargs, params=params)

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

  @property
  def training(self):
    return self.model.training

  def train_mode(self, mode=True):
    self.model.train(mode=mode)

  def eval_mode(self):
    self.model.eval()

  def epochs(self, num=None):
    """
    Yields current epoch count and keeps internal epoch count
    """
    for e in counter(start=self.num_epochs, end=self.num_epochs + num):
      yield e
      self.num_epochs += 1

  def batches(self, device=None):
    device = device or self.device

    for batch in self.loader:
      if self.transform_batch:
        batch = self.transform_batch(*batch)

      if device:
        yield self.num_steps, to(*batch, device=device)
      else:
        yield self.num_steps, batch

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

  def step(self, inputs=None, targets=None):
    """
    Single training step, including forward pass, backward and optimizer update
    """
    if not self.training:
      self.train_mode()

    outputs, loss = self.forward(
      inputs=inputs,
      targets=targets
    )

    self.update(
      loss=loss,
      inputs=inputs,
      targets=targets,
      outputs=outputs
    )

    return outputs, loss

  def forward(self, inputs=None, targets=None):
    outputs = self.model(inputs)
    loss = self.loss(outputs, targets)
    return outputs, loss

  def update(self, loss=None, inputs=None, targets=None, outputs=None):
    """
    Handles resetting gradients, running backward pass and optimizer step
    """
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def validate(self, loader=None, device=None):
    loader = loader or self.val_loader
    device = device or self.device

    if self.training:
      self.eval_mode()

    if self._use_callbacks:
      self.on_validation_start()

    ts, os, loss = [], [], None
    if loader is not None:
      with torch.no_grad():
        for inputs, targets, outputs in evaluate(
            model=self.model,
            batches=loader,
            device=device
        ):
          if self._use_callbacks:
            self.on_validation_batch(
              inputs=inputs,
              targets=targets,
              outputs=outputs
            )
          ts.append(targets)
          os.append(outputs)

        ts = torch.cat(ts)
        os = torch.cat(os)

        loss = self.loss(os, ts)

    if self._use_callbacks:
      self.on_validation_end(
        targets=ts,
        outputs=os,
        loss=loss
      )

    return loss

  def run(self, epochs=None):
    self._stop = False

    if self._use_callbacks:
      try:
        self.on_train_start()

        for epoch_idx in self.epochs(num=epochs):
          self.on_epoch_start(epoch=self.num_epochs)
          for batch_idx, (inputs, targets) in self.batches():
            self.on_batch_start(
              batch=batch_idx,
              inputs=inputs,
              targets=targets
            )
            try:
              outputs, loss = self.step(inputs=inputs, targets=targets)
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
              self._lr_scheduler_step(step=batch_idx)

            if self._stop: break
          if self._stop: break

          val_loss = self.validate() if self.val_loader else None
          if self.lr_scheduler and not self.lr_batch_step:
            self._lr_scheduler_step(
              step=epoch_idx,
              metric=self.history.metrics.running_mean('loss')
                if val_loss is None else val_loss,
            )

          self.on_epoch_end(epoch=epoch_idx)

        self.on_train_end()
      except Exception as e:
        self.on_error(e)
        raise e
    else:
      for epoch_idx in self.epochs(num=epochs):
        for batch_idx, (inputs, targets) in self.batches():
          outputs, loss = self.step(inputs=inputs, targets=targets)

          if self.lr_scheduler and self.lr_batch_step:
            self.lr_scheduler.step(epoch=batch_idx)

        val_loss = self.validate() if self.val_loader else None
        self._lr_scheduler_step(
          step=epoch_idx,
          metric=self.history.metrics.running_mean('loss') if val_loss is None else val_loss
        )

  def _lr_scheduler_step(self, step=None, metric=None):
    if self.lr_scheduler:
      if 'metrics' in self.lr_scheduler.step.__code__.co_varnames:
        self.lr_scheduler.step(metrics=metric, epoch=step)
      else:
        self.lr_scheduler(epoch=step)

  def __call__(self, *args, **kwargs):
    self.run(*args, **kwargs)

  def stop(self, val=True):
    self._stop = val

  def checkpoint(self, name=None) -> pathlib.Path:
    state = self.state_dict()
    path = self.paths.checkpoints / (
      f"{name}.th" if name else
      f"{timestr()}-epoch-{self.num_epochs:03d}-steps-{self.num_steps:05d}.th"
    )
    torch.save(state, str(path))
    return path

  def load_checkpoint(self, path, metadata=True, map_location=None):
    # TODO: add 'latest', 'best' support
    data = torch.load(path)
    self.load_state_dict(data, metadata=metadata, map_location=map_location)

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
          'class': fully_qualified_name(v)
        }

    return data

  def load_state_dict(self, data, metadata=True, map_location=None):
    """
    TODO: add a way to specify which parts should be loaded (ex: model only)
    """
    skipped = set()
    for k, v in data.items():
      if 'state_dict' in v and hasattr(self, k):
        getattr(self, k).load_state_dict(v['state_dict'], map_location=map_location)
        logging.debug(f"loaded {k}")
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
      f"\n  id={self.id},"
      f"\n  name={self.name},"
      f"\n  root={self.root},"
      f"\n  batch_size={self.batch_size},"
      f"\n  device={self.device}"
      "\n)"
    )