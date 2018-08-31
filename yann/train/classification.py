from typing import Optional

import torch
from torch.utils.data import DataLoader
import torchvision
import logging
import datetime
import pathlib
import numpy as np
import datetime


from .base import BaseTrainer
from yann.data import TransformDataset
from yann import resolve, evaluate
from yann import callbacks as yann_callbacks


def timestr():
  return f"{datetime.datetime.utcnow().strftime('%y-%m-%dT%H%M%S')}"


class Trainer(BaseTrainer):
  model: torch.nn.Module
  lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
  loader: DataLoader

  def __init__(
      self,
      model,
      dataset=None,
      optimizer=None,
      criterion=None,
      loader=None,
      sampler=None,
      collate=None,
      num_workers=8,
      transform=None,
      lr_scheduler=None,
      callbacks=None,
      device=None,
      trainable_parameters=None,
      batch_size=16,
      val_dataset=None,
      val_loader=None,
      val_transform=None,
      parallel=False,
      name=None):
    super().__init__()

    self.model = model
    if parallel:
      self.model = torch.nn.DataParallel(self.model)

    self.criterion = resolve(
      criterion,
      (torch.nn, torch.nn.functional),
      required=True,
      validate=lambda x: callable(x)
    )
    self.optimizer = resolve(
      optimizer,
      (torch.optim,),
      required=True,
      validate=lambda x: hasattr(x, 'step'),
      params=trainable_parameters or self.model.parameters()
    )
    self.dataset = resolve(
      dataset,
      (torchvision.datasets,),
      required=True,
    )

    if transform:
      self.dataset = TransformDataset(self.dataset, transform)

    self.batch_size = batch_size

    self.loader = loader or DataLoader(
      self.dataset,
      batch_size=self.batch_size,
      pin_memory=True,
      shuffle=False if sampler else True,
      sampler=sampler,
      #       collate_fn=collate,
      num_workers=num_workers
    )

    self.val_dataset = val_dataset

    val_transform = val_transform or transform
    if val_transform:
      self.val_dataset = TransformDataset(self.val_dataset, val_transform)

    self.val_loader = val_loader or (val_dataset and DataLoader(
      self.val_dataset,
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=num_workers
    ))

    self.lr_scheduler = resolve(
      lr_scheduler,
      (torch.optim.lr_scheduler,),
      optimizer=self.optimizer
    )

    self.num_samples = 0
    self.num_steps = 0
    self.num_epochs = 0

    self.name = name or (
      f"{timestr()}-{self.model.__class__.__name__}"
    )

    self.history = None
    has_history = False
    if callbacks:
      for c in callbacks:
        if isinstance(c, yann_callbacks.History):
          self.history = c
          has_history = True
          break
    self.history = self.history or yann_callbacks.History()

    self.callbacks = callbacks or [
      yann_callbacks.Logger()
    ]

    if not has_history:
      self.callbacks.append(self.history)

    self.function_callback = None

    if device:
      self.device = torch.device(device) if isinstance(device, str) else device
      self.to(self.device)
    else:
      self.device = None

  def to(self, device=None):
    self.device = device

    if self.model:
      self.model.to(self.device)

  def on(self, event, callback):
    if not self.function_callback:
      self.function_callback = yann_callbacks.FunctionCallback()
      self.callbacks.append(self.function_callback)

    self.function_callback.on(event, callback)


  def step(self, inputs, target):
    self.model.train()
    self.optimizer.zero_grad()

    outputs = self.model(inputs)
    loss = self.criterion(outputs, target)
    loss.backward()

    self.optimizer.step()

    self.num_steps += 1
    self.num_samples += len(inputs)

    return outputs, loss

  def batches(self):
    for inputs, targets in self.loader:
      yield inputs.to(self.device), targets.to(self.device)

  def validate(self, loader=None, device=None):
    loader = loader or self.val_loader
    device = device or self.device

    self.model.eval()

    self.on_validation_start()

    ts, os = [], []
    with torch.no_grad():
      for inputs, targets, outputs in evaluate(self.model, loader, device):
        self.on_validation_batch(inputs=inputs, targets=targets, outputs=outputs)
        ts.append(targets)
        os.append(outputs)

      ts = torch.cat(ts)
      os = torch.cat(os)
      loss = self.criterion(os, ts)
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
          if self._stop: break
        if self._stop: break

        if self.val_loader:
          val_loss = self.validate()
          if self.lr_scheduler:
            self.lr_scheduler.step(val_loss)
        elif self.lr_scheduler:
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

  def get_checkpoint_name(self):
    return (
      f"{self.name}-steps_{self.num_steps}.th"
    )

  @property
  def checkpoints_root(self):
    return pathlib.Path('./') / self.name / 'checkpoints'

  def checkpoint(self, root='./'):
    state = self.state_dict()
    path = pathlib.Path(root) / self.name / 'checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
      state,
      str(path / f"{timestr()}-epoch-{self.num_epochs:03d}.th")
    )

  def load_checkpoint(self, path, metadata=True):
    data = torch.load(path)
    self.load_state_dict(data, metadata=metadata)

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
