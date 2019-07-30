import logging

import datetime
import pathlib
import torch
import torch.nn
from torch.utils.data import DataLoader
from typing import Optional

from .. import callbacks as yann_callbacks
from .. import resolve, evaluate
from ..data import TransformDataset, get_dataset_name, Classes
from ..export import export
from ..inference.predict import Classifier
from ..train.base import BaseTrainer
from ..utils.decorators import lazy


def timestr(d=None):
  return f"{(d or datetime.datetime.utcnow()).strftime('%y-%m-%dT%H:%M:%S')}"


def get_model_name(model):
  if isinstance(model, torch.nn.DataParallel):
    model = model.module

  if hasattr(model, 'name'):
    return model.name

  return model.__class__.__name__


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
      lr_scheduler=None,
      lr_batch_step=False,
      callbacks=None,
      device=None,
      trainable_parameters=None,
      batch_size=16,
      val_dataset=None,
      val_loader=None,
      val_transform=None,
      classes=None,
      parallel=False,
      name=None,
      description=None,
      root='./train-runs/',
      metrics=None,
      collate=None
  ):
    super().__init__()

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
    self.optimizer = resolve.optimizer(
      optimizer,
      args=(trainable_parameters or self.model.parameters(),),
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

    self.batch_size = batch_size

    self.loader = loader or DataLoader(
      self.dataset,
      batch_size=self.batch_size,
      pin_memory=True,
      shuffle=False if sampler else True,
      sampler=sampler,
      num_workers=num_workers,
      **({'collate_fn': collate} if collate else {})
    )

    self.val_dataset = val_dataset

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

    self.num_samples = 0
    self.num_steps = 0
    self.num_epochs = 0

    self.time_created = datetime.datetime.utcnow()

    self.name = name or (
      f"{get_dataset_name(self.loader)}-{get_model_name(self.model)}"
    )
    self.description = description

    self.root = pathlib.Path(root) / self.name / timestr(self.time_created)
    self.root.mkdir(parents=True, exist_ok=True)

    self.history = None
    has_history = False
    if callbacks:
      for c in callbacks:
        if isinstance(c, yann_callbacks.History):
          self.history = c
          has_history = True
          break
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

    if device:
      self.device = torch.device(device) if isinstance(device, str) else device
      self.to(self.device)
    else:
      self.device = None

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
      if key == 'dataset' and hasattr(self, 'dataset') and self.dataset and hasattr(self, 'loader') and self.loader:
        raise ValueError(
          'Cannot modify dataset because a loader is defined '
          'and modifying dataset of a loader is not supported, '
          'try creating and setting a new loader instead'
        )

    logging.debug(f"setting '{key}' to {value}")
    super(Trainer, self).__setattr__(key, value)


  def to(self, device=None):
    self.device = device

    if self.model:
      self.model.to(self.device)

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

  def step(self, inputs, target):
    self.model.train()
    self.optimizer.zero_grad()

    outputs = self.model(inputs)
    loss = self.loss(outputs, target)
    loss.backward()
    self.optimizer.step()

    self.num_steps += 1
    self.num_samples += len(inputs)

    return outputs, loss

  def batches(self):
    for inputs, targets in self.loader:
      if self.device:
        yield inputs.to(self.device), targets.to(self.device)
      else:
        yield inputs, targets

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

  def run(self, epochs=1):
    self._stop = False
    try:
      self.on_train_start()

      for _ in range(epochs):
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

          self.on_batch_end(batch=batch_idx, inputs=inputs, targets=targets,
                            outputs=outputs, loss=loss)

          if self.lr_scheduler and self.lr_batch_step:
            self.lr_scheduler.step()

          if self._stop: break
        if self._stop: break

        if self.val_loader:
          val_loss = self.validate()
          if self.lr_scheduler and not self.lr_batch_step:
            self.lr_scheduler.step(val_loss)
        elif self.lr_scheduler and not self.lr_batch_step:
          self.lr_scheduler.step()

        self.on_epoch_end(epoch=self.num_epochs)
        self.num_epochs += 1

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

  def get_checkpoint_name(self):
    return (
      f"{self.name}-steps_{self.num_steps}.th"
    )

  @property
  def checkpoints_root(self):
    root = self.root / 'checkpoints'
    root.mkdir(parents=True, exist_ok=True)
    return root

  def checkpoint(self, root='./', name=None):
    state = self.state_dict()
    path = str(self.checkpoints_root /
               (f"{name}.th" if name else
                f"{timestr()}-epoch-{self.num_epochs:03d}-"
                f"steps-{self.num_steps:05d}.th"))
    torch.save(state, path)
    return path

  def load_checkpoint(self, path, metadata=True):
    data = torch.load(path)
    self.load_state_dict(data, metadata=metadata)

  def export(self, path=None, trace=False, meta=None, postprocess=None):
    path = path or self.root / 'exports' / timestr()
    export(
      model=self.model,
      preprocess=self.val_transform,
      postprocess=postprocess,
      classes=self.classes,
      trace=trace and next(iter(self.loader)),
      path=path,
      meta=dict(
        name=self.name,
        root=str(self.root),
        dataset=get_dataset_name(self.dataset),
        num_steps=self.num_steps,
        **(meta or {})
      )
    )

    return path

  def state_dict(self):
    data = {
      'metadata': {
        'num_steps': self.num_steps,
        'num_samples': self.num_samples,
        'num_epochs': self.num_epochs,
        'batch_size': self.batch_size,
        'name': self.name,
        'time_created': datetime.datetime.utcnow()
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

  def __str__(self):
    return f"""
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


PROGRESS
========
epochs: {self.num_epochs}
steps: {self.num_steps}
samples: {self.num_samples}"""
