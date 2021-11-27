import inspect

import datetime
import logging
import torch
import torch.nn
import types
from pathlib import Path
from torch.optim.optimizer import Optimizer
from torch.utils.data import Sampler, DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Callable

import yann
from yann import resolve, evaluate, to, default
from yann.utils import fully_qualified_name
from yann.params import HyperParams
from yann.callbacks import get_callbacks, Logger
from yann.callbacks.callbacks import Callbacks
from yann.data import get_dataset_name, Classes
from yann.data.io import save_json
from yann.datasets import TransformDataset
from yann.distributed import Dist
from yann.export import export
from yann.train.base import BaseTrainer
from yann.train.paths import Paths
from yann.utils import counter, timestr, hash_params
from yann.utils.ids import memorable_id


class Trainer(BaseTrainer):
  """
  def train():
    for epoch in self.epochs:
      for batch in self.batches:
        self.step(batch)
      self.validate()

  def step():
    self.forward()
    self.update()

  """
  params: Optional[HyperParams] = None
  model: Optional[torch.nn.Module] = None
  loss: Optional[Callable] = None

  optimizer: Optional[Optimizer] = None
  lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
  clip_grad: Optional = None

  loader: Optional[DataLoader] = None
  classes: Optional[Classes] = None
  sampler: Optional[Sampler] = None

  paths: Paths = None
  callbacks: Optional[Callbacks] = None
  log: Optional[Logger] = None

  # automatic mixed precision
  grad_scaler: Optional[GradScaler] = None

  history = None

  DataLoader = DataLoader

  def __init__(
      self,
      model=None,
      dataset=None,
      optimizer=None,
      loss=None,
      loader=None,
      sampler=None,
      batch_sampler=None,
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
      parallel=None,
      amp=False,
      grad_scaler=None,
      name=None,
      description=None,
      root='./experiments/train-runs/',
      metrics=None,
      collate=None,
      params=None,
      pin_memory=True,
      step=None,
      id=None,
      dist=None,
      benchmark=True,
      jit='script',
  ):
    super().__init__()
    self.id = id or memorable_id()
    self.params = params

    self.num_samples = 0
    self.num_steps = 0
    self.num_epochs = 0

    self.time_created = datetime.datetime.utcnow()

    if benchmark:
      yann.benchmark()

    self.dist = dist or Dist()
    self.dist.initialize()
    if self.dist.is_enabled:
      device = device or self.dist.device

    device = default.device if device is None else device
    self.device = torch.device(device) if isinstance(device, str) else device

    self.model = resolve.model(model, required=False, validate=callable)

    if jit:
      self.model = torch.jit.script(self.model)
    self.loss = resolve.loss(loss, required=False, validate=callable)

    self.parallel = parallel
    self._init_parallel(parallel)

    self._init_optim(
      parameters=parameters,
      optimizer=optimizer,
      lr_scheduler=lr_scheduler,
      lr_batch_step=lr_batch_step
    )

    self._init_data_loaders(
      dataset=dataset,
      classes=classes,
      transform=transform,
      transform_batch=transform_batch,
      batch_size=batch_size,
      sampler=sampler,
      batch_sampler=batch_sampler,

      loader=loader,
      pin_memory=pin_memory,
      num_workers=num_workers,
      collate=collate,

      val_dataset=val_dataset,
      val_transform=val_transform,
      val_loader=val_loader,
    )

    self._init_amp(amp=amp, grad_scaler=grad_scaler)
    self._init_callbacks(callbacks=callbacks, metrics=metrics)

    if step is not None:
      self.override(self.step, step)

    self.to(self.device)

    self.name = name or (
      f"{get_dataset_name(self.loader)}-{yann.get_model_name(self.model)}"
    )
    self.description = description

    self.paths = Paths(Path(root) / self.name / timestr(self.time_created))
    self.paths.create()

  def _init_callbacks(self, callbacks, metrics):
    if not self.dist.is_main:
      # NOTE: disable most callbacks if not on main
      self.history = yann.callbacks.History(*(metrics or ()))
      self.callbacks = Callbacks(self.history)
      return

    self.callbacks = Callbacks(get_callbacks() if callbacks is True else callbacks)

    if 'history' not in self.callbacks:
      metrics = (metrics,) if isinstance(metrics, str) else metrics
      self.callbacks.history = yann.callbacks.History(*(metrics or ()))

    self.history = self.callbacks.history
    self.callbacks.move_to_start('history')

  def _init_parallel(self, parallel):
    if self.model is not None:
      if parallel == 'dp':
        if not isinstance(self.model, torch.nn.parallel.DataParallel):
          self.model = torch.nn.DataParallel(self.model)
      elif parallel == 'ddp' or parallel is None and self.dist.is_enabled:
        if not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
          self.model.to(self.device)
          self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.dist.local_rank],
            output_device=self.dist.local_rank,
          )

  def _init_data_loaders(
      self,
      dataset=None,
      loader=None,
      transform=None,
      classes=None,
      transform_batch=None,
      sampler=None,
      batch_sampler=None,
      batch_size=None,
      pin_memory=None,
      collate=None,
      num_workers=None,
      prefetch_factor=2,
      persistent_workers=True,
      val_dataset=None,
      val_transform=None,
      val_loader=None,
  ):
    self.dataset = resolve.dataset(dataset, required=False)

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

    if not sampler and self.dist.is_enabled:
      sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)

    if loader is not None:
      self.loader = loader
    elif self.dataset is not None:
      if batch_sampler is not None:
        loader_signature = inspect.signature(self.DataLoader)
        if 'batch_sampler' not in loader_signature.parameters:
          raise ValueError(
            'batch_sampler provided but DataLoader does not support it, might need to upgrade pytorch to newer version')
        self.loader = self.DataLoader(
          dataset=self.dataset,
          batch_sampler=batch_sampler,
          pin_memory=pin_memory,
          num_workers=num_workers,
          persistent_workers=persistent_workers,
          prefetch_factor=prefetch_factor,
          **({'collate_fn': collate} if collate else {})
        )
      else:
        self.loader = self.DataLoader(
          dataset=self.dataset,
          batch_size=self.batch_size,
          pin_memory=pin_memory,
          shuffle=False if sampler else True,
          sampler=sampler,
          num_workers=num_workers,
          persistent_workers=persistent_workers,
          prefetch_factor=prefetch_factor,
          **({'collate_fn': collate} if collate else {})
        )

    self.val_dataset = resolve.dataset(
      val_dataset,
      required=False,
    )

    self.val_transform = val_transform or transform
    if self.val_transform:
      self.val_dataset = TransformDataset(self.val_dataset, self.val_transform)

    self.val_loader = val_loader or (val_dataset and self.DataLoader(
      self.val_dataset,
      batch_size=batch_size,
      shuffle=False,
      pin_memory=pin_memory,
      num_workers=num_workers
    ))

  def _init_optim(
      self,
      parameters='trainable',
      optimizer=None,
      lr_scheduler=None,
      lr_batch_step=None,
      clip_grad=None
  ):
    if parameters == 'trainable' and self.model:
      parameters = yann.trainable(self.model.parameters())

    self.optimizer = resolve.optimizer(
      optimizer,
      args=(parameters or self.model.parameters(),),
      required=False,
      validate=lambda x: hasattr(x, 'step')
    )

    self.lr_scheduler = resolve.lr_scheduler(
      lr_scheduler,
      kwargs=dict(optimizer=self.optimizer)
    )
    self.lr_batch_step = lr_batch_step
    self.clip_grad = clip_grad

  def _init_amp(self, amp, grad_scaler):
    self.amp = amp
    if grad_scaler is False:
      self.grad_scaler = None
    else:
      self.grad_scaler = grad_scaler or (GradScaler() if self.amp else None)

    self.clip_grad = None

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
    if self.callbacks is not None:
      return self.callbacks.on(event, callback)
    logging.warning('.on() callback registration ignored because callbacks are not defined')

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
        yield to(*batch, device=device)
      else:
        yield batch

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
    with autocast(enabled=self.amp):
      outputs = self.model(inputs)
      loss = self.loss(outputs, targets)
    return outputs, loss

  def update(self, loss=None, inputs=None, targets=None, outputs=None):
    """
    Handles resetting gradients, running backward pass and optimizer step
    """
    self.optimizer.zero_grad()

    if self.grad_scaler:
      self.grad_scaler.scale(loss).backward()
      self.grad_scaler.step(self.optimizer)

      if self.clip_grad:
        self.grad_scaler._unscale(self.optimizer)
        self.clip_grad(self.model.parameters())
      self.grad_scaler.update()
    else:
      loss.backward()

      if self.clip_grad:
        self.clip_grad(self.model.parameters())

      self.optimizer.step()

  def validate(self, loader=None, device=None):
    loader = loader or self.val_loader
    device = device or self.device

    if self.training:
      self.eval_mode()

    if self.callbacks:
      self.callbacks.on_validation_start(trainer=self)

    ts, os, loss = [], [], None
    if loader is not None:
      with torch.inference_mode():
        for inputs, targets, outputs in evaluate(
            model=self.model,
            batches=loader,
            device=device
        ):
          if self.callbacks:
            self.callbacks.on_validation_batch(
              inputs=inputs,
              targets=targets,
              outputs=outputs,
              trainer=self
            )
          ts.append(targets)
          os.append(outputs)

        ts = torch.cat(ts)
        os = torch.cat(os)

        loss = self.loss(os, ts)

    if self.callbacks:
      self.callbacks.on_validation_end(
        targets=ts,
        outputs=os,
        loss=loss,
        trainer=self
      )

    return loss

  def run(self, epochs=None):
    self._stop = False

    if self.callbacks:
      try:
        self.callbacks.on_train_start(trainer=self)

        for epoch_idx in self.epochs(num=epochs):
          self.callbacks.on_epoch_start(
            epoch=self.num_epochs,
            trainer=self
          )
          for inputs, targets in self.batches():
            self.callbacks.on_step_start(
              index=self.num_steps,
              inputs=inputs,
              targets=targets,
              trainer=self
            )
            try:
              outputs, loss = self.step(inputs=inputs, targets=targets)
            except KeyboardInterrupt as e:
              self.stop()
              break
            except Exception as e:
              self.callbacks.on_step_error(
                index=self.num_steps,
                error=e,
                trainer=self
              )
              if self._stop: break
              raise e

            self.callbacks.on_step_end(
              index=self.num_steps,
              inputs=inputs,
              targets=targets,
              outputs=outputs,
              loss=loss,
              trainer=self
            )

            if self.lr_scheduler and self.lr_batch_step:
              self._lr_scheduler_step(step=self.num_steps)

            self.num_steps += 1
            self.num_samples += len(inputs)

            if self._stop: break
          if self._stop: break

          val_loss = self.validate() if self.val_loader else None
          if self.lr_scheduler and not self.lr_batch_step:
            self._lr_scheduler_step(
              step=epoch_idx,
              metric=self.history.metrics.running_mean('loss')
                if val_loss is None else val_loss,
            )

          self.callbacks.on_epoch_end(epoch=epoch_idx, trainer=self)

        self.callbacks.on_train_end(trainer=self)
      except KeyboardInterrupt as e:
        self.stop()
      except Exception as e:
        self.callbacks.on_error(e, trainer=self)
        raise e
    else:
      for epoch_idx in self.epochs(num=epochs):
        for inputs, targets in self.batches():
          outputs, loss = self.step(inputs=inputs, targets=targets)

          if self.lr_scheduler and self.lr_batch_step:
            self.lr_scheduler.step(epoch=self.num_steps)

          self.num_steps += 1
          self.num_samples += len(inputs)

        val_loss = self.validate() if self.val_loader else None
        self._lr_scheduler_step(
          step=epoch_idx,
          metric=self.history.metrics.running_mean('loss')
          if val_loss is None else val_loss
        )

  def _lr_scheduler_step(self, step=None, metric=None):
    if self.lr_scheduler:
      args = dict()
      if 'metrics' in self.lr_scheduler.step.__code__.co_varnames:
        args['metrics'] = metric
      if 'metrics' in self.lr_scheduler.step.__code__.co_varnames:
        args['epoch'] = step
      self.lr_scheduler.step(**args)

  def __call__(self, *args, **kwargs):
    self.run(*args, **kwargs)

  def stop(self, val=True):
    self._stop = val

  def checkpoint(self, name=None) -> Path:
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