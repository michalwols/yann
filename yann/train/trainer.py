import datetime
import inspect
import logging
import types
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Union

import torch
import torch.nn
from torch.amp import GradScaler, autocast
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Sampler
from typing_extensions import Literal, Unpack

import yann
import yann.distributed
from yann.data import Classes, get_dataset_name
from yann.datasets import Subset, TransformDataset
from yann.distributed import Dist
from yann.export import export
from yann.train.base import BaseTrainer
from yann.train.paths import Paths
from yann.utils import (
  apply_known,
  counter,
  fully_qualified_name,
  hash_params,
  memorable_id,
  timestr,
)
from yann.utils.bash import git_diff, pip_freeze
from yann.utils.timer import time

log = logging.getLogger(__name__)


class Keys:
  """
  keys for data batch
  """

  ids = None
  inputs = 0
  targets = 1


class TrainState:
  num_steps: int = 0
  num_epochs: int = 0
  num_samples: int = 0


class Params(yann.params.HyperParams):
  id: Union[str, int, None] = None
  name: Union[str, None] = None
  description: Optional[str] = None
  project: Optional[str] = None
  meta: Optional[Dict] = None

  # root directory where training runs are stored
  root: Union[str, Path, Paths] = './runs/'

  model: Union[torch.nn.Module, str, None] = None

  dataset: Union[torch.utils.data.Dataset, str, None] = None
  subset: Optional[int] = None
  batch_size: int = None
  classes: Union[yann.data.Classes, Sequence[str], None] = None

  optimizer: Union[torch.optim.Optimizer, str, None] = None
  parameters: Union[torch.nn.ParameterList, Literal['trainable']] = 'trainable'
  lr: Optional[float] = None
  weight_decay: Optional[float] = None
  momentum: Optional[float] = None
  lr_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, None] = None
  lr_batch_step: bool = False
  none_grad: bool = True

  epochs: Optional[int] = None

  # warmup
  # min_lr
  # max_lr

  loss: Union[torch.nn.Module, str, None] = None

  loader: Union[torch.utils.data.DataLoader, None] = None
  num_workers: int = 8
  collate: Union[Callable, None] = None
  pin_memory: bool = True
  sampler: Union[torch.utils.data.Sampler, None] = None
  batch_sampler: Union[torch.utils.data.BatchSampler, None] = None
  prefetch_factor: Optional[int] = 2
  persistent_workers: Optional[bool] = True

  transform: Union[
    Callable,
    Mapping[str, Callable],
    Sequence[Callable],
    None,
  ] = None
  transform_batch: Union[Callable, None] = None

  callbacks: Union[
    Sequence['yann.callbacks.Callback'],
    'yann.callbacks.Callbacks',
    None,
  ] = None
  device: Union[torch.device, str, None] = None
  dtype: Optional[torch.dtype] = None

  val_dataset: Union[torch.utils.data.Dataset, float, str, None] = None
  val_subset: Optional[int] = None
  val_loader: Union[torch.utils.data.DataLoader, None] = None
  val_transform: Union[
    Callable,
    Mapping[str, Callable],
    Sequence[Callable],
    None,
  ] = None

  metrics: Union[
    Dict[str, Callable],
    Sequence[Callable],
    Sequence[str],
    None,
  ] = None

  dist: Optional[Dist] = None
  parallel: Union[None, Literal['dp', 'ddp']] = None

  amp: bool = False
  grad_scaler: Optional[torch.amp.GradScaler] = None

  benchmark: bool = True
  jit: bool = False
  memory_format: Optional[str] = 'preserve_format'
  aot_autograd: bool = False
  cuda_graph: bool = False

  compile: bool = False
  tf32: bool = False

  step: Optional[Callable] = None
  place: Optional[Union[Callable, dict, tuple, yann.data.place.Place]] = None
  clip_grad: Union[Callable, 'yann.optim.clip.GradClipper', dict] = None
  seed: Optional[int] = None

  from_checkpoint: Optional[str] = None


class Trainer(TrainState, BaseTrainer):
  """
  A training loop wrapper class that provides convenient initialization,
  loading and checkpointing



  def train():
    for epoch in self.epochs():
      for batch in self.batches():
        self.step(batch)
      self.validate()

  def step():
    self.forward()
    self.update()

  """

  Params = Params

  params: Params
  model: Optional[torch.nn.Module] = None
  loss: Optional[Callable] = None

  optimizer: Optional[Optimizer] = None
  lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
  clip_grad: Optional = None

  loader: Optional[DataLoader] = None
  classes: Optional[Classes] = None
  sampler: Optional[Sampler] = None

  paths: Paths = None
  callbacks: Optional['yann.callbacks.Callbacks'] = None
  log: Optional['yann.callbacks.Logger'] = None

  # automatic mixed precision
  grad_scaler: Optional[GradScaler] = None

  history: 'yann.callbacks.History' = None

  # Dict with training run summary information
  summary: dict

  keys = Keys()

  DataLoader = DataLoader

  @classmethod
  def from_params(cls, params: Params, **kwargs: Unpack[Params]):
    return cls(**{**params, **kwargs}, params=params)

  @time('Initialize Trainer')
  def __init__(
    self,
    /,
    params: Union[Params, str, None] = None,
    **kwargs: Unpack[Params],
  ):
    super().__init__()

    self.params = (
      params
      if isinstance(params, self.Params)
      else self.Params(params) if params else self.Params()
    )
    self.params.update(kwargs)

    if self.params.seed is not None:
      yann.seed(self.params.seed)

    self.id = self.params.id or memorable_id()
    self.name = self.params.name
    self.description = self.params.description
    self.project = self.params.project
    self.meta = self.params.meta
    self.summary = {}

    self.time_created = datetime.datetime.utcnow()

    self._epochs = self.params.epochs

    if self.params.benchmark:
      yann.benchmark()

    self.dist = self.params.dist or Dist()
    self.dist.initialize()

    device = self.params.device
    if self.dist.is_enabled:
      device = device or self.dist.device

    device = yann.default.device if device is None else device
    self.device = torch.device(device) if isinstance(device, str) else device

    self.memory_format = yann.memory_formats.get(self.params.memory_format, self.params.memory_format)
    self.dtype = self.params.dtype

    self.lr_batch_step = self.params.lr_batch_step
    self.none_grad = self.params.none_grad

    self.model = yann.resolve.model(self.params.model, required=False, validate=callable)

    if self.params.tf32:
      torch.backends.cuda.matmul.allow_tf32 = True

    compile_arg = self.params.compile
    if compile_arg:
      if isinstance(compile_arg, dict):
        self.model = torch.compile(self.model, **compile_arg)
      else:
        self.model = torch.compile(self.model)

    if self.params.jit:
      self.model = torch.jit.script(self.model)
    if self.params.aot_autograd:
      try:
        from functorch.compile import memory_efficient_fusion
      except ImportError:
        raise ValueError('functorch must be installed for aot_autograd support')
      self.model = memory_efficient_fusion(self.model)

    self.loss = yann.resolve.loss(self.params.loss, required=False, validate=callable)

    self._init_parallel()
    self._init_optim()
    self._init_data_loaders()
    self._init_amp()
    self._init_callbacks()

    if self.params.step is not None:
      self.override(self.step, self.params.step)

    if self.params.place is not None:
      if isinstance(self.params.place, Callable):
        self.place = self.params.place
      else:
        from yann.data.place import Place
        self.place = Place(self.params.place)

    self.to(device=self.device, memory_format=self.memory_format)

    self.name = self.name or (
      f'{get_dataset_name(self.loader)}-{yann.get_model_name(self.model)}'
    )

    root = self.params.root
    if isinstance(root, Paths):
      self.paths = root
    else:
      self.paths = Paths(
        Path(root or yann.default.train_root) / self.name / timestr(self.time_created),
      )
    self.paths.create()

    if self.dist.is_main:
      try:
        yann.save.txt(git_diff(), self.paths.git_diff)
      except:
        pass  # not in git repo
      yann.save.txt(pip_freeze(), self.paths.requirements)

    if self.params.from_checkpoint:
      self.load_checkpoint(self.params.from_checkpoint)

    self.update_summary()
    self.save_summary()

  def _init_callbacks(self, **kwargs):
    from yann.callbacks import get_callbacks
    from yann.callbacks.callbacks import Callbacks

    callbacks = get_callbacks() if self.params.callbacks is True else self.params.callbacks
    callbacks = callbacks or []
    callbacks = [
      c for c in callbacks if yann.distributed.matches(c.dist_placement, self.dist)
    ]

    self.callbacks = Callbacks(callbacks)

    if 'history' not in self.callbacks:
      metrics = self.params.metrics
      metrics = (metrics,) if isinstance(metrics, str) else metrics
      self.callbacks.history = (
        yann.callbacks.History(**metrics)
        if isinstance(metrics, dict)
        else yann.callbacks.History(*metrics or ())
      )

    self.history = self.callbacks.history
    self.callbacks.move_to_start('history')

  def _init_parallel(self, **kwargs):
    if self.model is not None:
      if self.params.parallel == 'dp':
        if not isinstance(self.model, torch.nn.parallel.DataParallel):
          self.model = torch.nn.DataParallel(self.model)
      elif self.params.parallel == 'ddp' or (self.params.parallel is None and self.dist.is_enabled):
        self.parallel = 'ddp'  # Store the effective parallel type
        if not isinstance(
          self.model,
          torch.nn.parallel.DistributedDataParallel,
        ):
          self.model.to(self.device)
          self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.dist.local_rank],
            output_device=self.dist.local_rank,
            find_unused_parameters=yann.default.ddp_find_unused_parameters,
          )

  @time('Initialize Data Loading')
  def _init_data_loaders(self, **kwargs):
    self.dataset = yann.resolve.dataset(self.params.dataset, required=False)
    if self.dataset and self.params.subset is not None:
      self.dataset = yann.datasets.Subset(
        self.dataset,
        *self.params.subset if isinstance(self.params.subset, tuple) else (self.params.subset,),
      )

    val_dataset = self.params.val_dataset # Temporary for split logic
    if isinstance(val_dataset, float):
      split = 1 - val_dataset
      self.val_dataset_resolved = Subset(self.dataset, split, 1.0) # Resolved val dataset
      self.dataset = Subset(self.dataset, 0, split) # Updated train dataset
    else:
      self.val_dataset_resolved = yann.resolve.dataset(val_dataset, required=False)

    classes = self.params.classes
    if classes:
      self.classes = classes if isinstance(classes, Classes) else Classes(classes)
    elif hasattr(self.dataset, 'classes') and isinstance(
      self.dataset.classes,
      Classes,
    ):
      self.classes = self.dataset.classes
    else:
      self.classes = None

    if self.params.transform:
      self.dataset = TransformDataset(self.dataset, self.params.transform)

    self.sampler = self.params.sampler
    if not self.sampler and self.dist.is_enabled and self.dataset:
      self.sampler = torch.utils.data.distributed.DistributedSampler(
        self.dataset,
        num_replicas=self.dist.world_size,
        rank=self.dist.rank,
      )

    if self.params.loader is not None:
      self.loader = self.params.loader
    elif self.dataset is not None:
      if self.params.batch_sampler is not None:
        loader_signature = inspect.signature(self.DataLoader)
        if 'batch_sampler' not in loader_signature.parameters:
          raise ValueError(
            'batch_sampler provided but DataLoader does not support it, might need to upgrade pytorch to newer version',
          )
        self.loader = self.DataLoader(
          dataset=self.dataset,
          batch_sampler=self.params.batch_sampler,
          pin_memory=self.params.pin_memory,
          num_workers=self.params.num_workers,
          persistent_workers=self.params.persistent_workers and self.params.num_workers > 0,
          prefetch_factor=self.params.prefetch_factor,
          **({'collate_fn': self.params.collate} if self.params.collate else {}),
        )
      else:
        self.loader = self.DataLoader(
          dataset=self.dataset,
          batch_size=self.params.batch_size or yann.default.batch_size,
          pin_memory=self.params.pin_memory,
          shuffle=False if self.sampler else True,
          sampler=self.sampler,
          num_workers=self.params.num_workers,
          persistent_workers=self.params.persistent_workers and self.params.num_workers > 0,
          prefetch_factor=self.params.prefetch_factor,
          **({'collate_fn': self.params.collate} if self.params.collate else {}),
        )

    if self.val_dataset_resolved and self.params.val_subset is not None:
      self.val_dataset_resolved = yann.datasets.Subset(
        self.val_dataset_resolved,
        *self.params.val_subset if isinstance(self.params.val_subset, tuple) else (self.params.val_subset,),
      )

    resolved_val_transform = self.params.val_transform or self.params.transform
    if resolved_val_transform and self.val_dataset_resolved:
        self.val_dataset_resolved = TransformDataset(self.val_dataset_resolved, resolved_val_transform)

    self.val_loader = self.params.val_loader or (
      self.val_dataset_resolved
      and self.DataLoader(
        self.val_dataset_resolved,
        batch_size=self.params.batch_size or yann.default.batch_size,
        shuffle=False,
        pin_memory=self.params.pin_memory,
        num_workers=self.params.num_workers,
      )
    )

  def _init_optim(self, **kwargs):
    parameters = self.params.parameters
    if parameters == 'trainable' and self.model:
      parameters = yann.trainable(self.model.parameters())

    self.optimizer = yann.resolve.optimizer(
      self.params.optimizer,
      args=(parameters or (self.model.parameters() if self.model else None),),
      kwargs={
        k: v
        for k, v in dict(
          lr=self.params.lr,
          weight_decay=self.params.weight_decay,
          momentum=self.params.momentum,
        ).items()
        if v is not None
      },
      required=False,
      validate=lambda x: hasattr(x, 'step'),
    )

    self.lr_scheduler = yann.resolve.lr_scheduler(
      self.params.lr_scheduler,
      kwargs=dict(optimizer=self.optimizer),
    )

    self.clip_grad = None
    if self.params.clip_grad:
      from yann.optim import GradClipper

      if self.params.clip_grad is True:
        self.clip_grad = GradClipper(value=1)
      elif isinstance(self.params.clip_grad, dict):
        self.clip_grad = GradClipper(**self.params.clip_grad)
      elif callable(self.params.clip_grad):
        self.clip_grad = self.params.clip_grad

  def _init_amp(self, **kwargs):
    if self.params.grad_scaler is False:
      self.grad_scaler = None
    else:
      self.grad_scaler = self.params.grad_scaler or (GradScaler() if self.params.amp else None)

  @property
  def root(self):
    """for backwards compatibility, self.paths.root used to be on self.root"""
    return self.paths.root

  def __setattr__(self, key, value):
    if key == 'optimizer':
      if hasattr(self, 'lr_scheduler') and hasattr(
        self.lr_scheduler,
        'optimizer',
      ):
        self.lr_scheduler.optimizer = value
    if key == 'loader':
      if hasattr(self, 'dataset') and hasattr(value, 'dataset'):
        super(Trainer, self).__setattr__('dataset', value.dataset)
      if hasattr(value, 'batch_size'):
        pass # batch_size is now only on params
    if key == 'batch_size':
      if self.params.batch_size != value:
        if hasattr(self, 'loader') and self.loader:
          raise ValueError(
            'Cannot modify batch_size because a loader is defined '
            'and modifying batch size of a loader is not supported, '
            'try creating and setting a new loader instead',
          )
      if (
        key == 'dataset'
        and hasattr(self, 'dataset')
        and self.dataset
        and hasattr(self, 'loader')
        and self.loader
      ):
        raise ValueError(
          'Cannot modify dataset because a loader is defined '
          'and modifying dataset of a loader is not supported, '
          'try creating and setting a new loader instead',
        )

    log.debug(f"setting '{key}' to {value}")
    super(Trainer, self).__setattr__(key, value)

  def to(self, **kwargs):
    """
    Places model, loss and optimizer on device
    """
    self.device = kwargs.pop('device', None) or self.device
    self.memory_format = kwargs.pop('memory_format', None) or self.memory_format
    yann.to(
      (self.model, self.loss, self.optimizer),
      device=self.device,
      memory_format=self.memory_format,
      dtype=self.dtype,
      **kwargs,
    )
    return self

  def place(self, batch, **kwargs):
    """
    Places batch on device
    """
    self.device = kwargs.pop('device', None) or self.device
    self.memory_format = kwargs.pop('memory_format', None) or self.memory_format

    # FIXME: find better way to handle channels last for specific entries in batch
    # possibly let memory_format take a dict or list to match batch
    if self.memory_format == torch.channels_last:
      if self.keys.inputs is not None:
        return (
          yann.to(
            batch[self.keys.inputs],
            device=self.device,
            memory_format=self.memory_format,
            **kwargs,
          ),
          *yann.to(batch[1:], device=self.device, **kwargs),
        )

    return yann.to(
      batch,
      device=self.device,
      memory_format=self.memory_format,
      **kwargs,
    )

  def on(self, event, callback=None):
    if self.callbacks is not None:
      return self.callbacks.on(event, callback)
    log.warning(
      '.on() callback registration ignored because callbacks are not defined',
    )

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
    for batch in self.loader:
      if self.params.transform_batch:
        batch = self.params.transform_batch(*batch)

      yield self.place(batch, device=device)

  def override(self, method, function: Union[bool, Callable] = False):
    """
    Override a method of the trainer
    Args:
      method: str or method reference
      function: function to be used as a replacement for the given method
    """
    method = method if isinstance(method, str) else method.__name__
    if not hasattr(self, method):
      raise AttributeError(
        f"Can't override method '{method}' because it's not defined",
      )
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

    outputs, loss = self.forward(inputs=inputs, targets=targets)

    self.update(loss=loss, inputs=inputs, targets=targets, outputs=outputs)

    return outputs, loss

  def forward(self, inputs=None, targets=None):
    with autocast(
      device_type=self.device.type,
      dtype=self.dtype,
      enabled=self.params.amp,
    ):
      outputs = self.model(inputs)
      if self.loss:
        loss = self.loss(outputs, targets)
        return outputs, loss
      else:
        return outputs, outputs  # FIXME: return None?

  def update(self, loss=None, inputs=None, targets=None, outputs=None):
    """
    Handles resetting gradients, running backward pass and optimizer step
    """

    # TODO: add gradient accumulation

    self.optimizer.zero_grad(set_to_none=self.none_grad)

    if self.grad_scaler:
      self.grad_scaler.scale(loss).backward()
      self.grad_scaler.step(self.optimizer)

      if self.clip_grad:
        self.grad_scaler.unscale_(self.optimizer)
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
        for inputs, targets, outputs in yann.evaluate(
          model=self.model,
          batches=loader,
          device=device,
        ):
          if self.callbacks:
            self.callbacks.on_validation_batch(
              inputs=inputs,
              targets=targets,
              outputs=outputs,
              trainer=self,
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
        trainer=self,
      )

    return loss

  def run(self, epochs=None):
    epochs = epochs or self._epochs

    self._stop = False

    if self.callbacks:
      try:
        self.callbacks.on_train_start(trainer=self)

        for epoch_idx in self.epochs(num=epochs):
          self.callbacks.on_epoch_start(epoch=self.num_epochs, trainer=self)

          if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch_idx)

          for batch in self.batches():
            if isinstance(batch, dict):
              inputs, targets = batch, batch  # Pass dict as both inputs and targets
            else:
              inputs, targets = batch  # Traditional tuple unpacking
            
            self.callbacks.on_step_start(
              index=self.num_steps,
              inputs=inputs,
              targets=targets,
              trainer=self,
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
                trainer=self,
              )
              if self._stop:
                break
              raise e

            self.callbacks.on_step_end(
              index=self.num_steps,
              inputs=inputs,
              targets=targets,
              outputs=outputs,
              loss=loss,
              trainer=self,
            )

            if self.lr_scheduler and self.lr_batch_step:
              self._lr_scheduler_step(step=self.num_steps)

            self.num_steps += 1
            self.num_samples += len(inputs)

            if self._stop:
              break
          if self._stop:
            break

          val_loss = self.validate() if self.val_loader else None
          if self.lr_scheduler and not self.lr_batch_step:
            self._lr_scheduler_step(
              step=epoch_idx,
              metric=self.history.metrics.running_mean('loss')
              if val_loss is None
              else val_loss,
            )

          self.callbacks.on_epoch_end(epoch=epoch_idx, trainer=self)

        self.callbacks.on_train_end(trainer=self)
      except KeyboardInterrupt as e:
        self.stop()
      except Exception as e:
        self.callbacks.on_error(e, trainer=self)
        raise e
      finally:
        self.update_summary()
        self.save_summary()
    else:
      for epoch_idx in self.epochs(num=epochs):
        if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
          self.sampler.set_epoch(epoch_idx)

        for batch in self.batches():
          if isinstance(batch, dict):
            inputs, targets = batch, batch  # Pass dict as both inputs and targets
          else:
            inputs, targets = batch  # Traditional tuple unpacking
          
          outputs, loss = self.step(inputs=inputs, targets=targets)

          if self.lr_scheduler and self.lr_batch_step:
            self.lr_scheduler.step(epoch=self.num_steps)

          self.num_steps += 1
          self.num_samples += len(inputs) if not isinstance(inputs, dict) else len(next(iter(inputs.values())))

        val_loss = self.validate() if self.val_loader else None
        self._lr_scheduler_step(
          step=epoch_idx,
          metric=self.history.metrics.running_mean('loss')
          if val_loss is None
          else val_loss,
        )
      self.update_summary()
      self.save_summary()

  def _lr_scheduler_step(self, step=None, metric=None):
    if self.lr_scheduler:
      args = dict()
      if 'metrics' in self.lr_scheduler.step.__code__.co_varnames:
        args['metrics'] = metric
      # if 'epoch' in self.lr_scheduler.step.__code__.co_varnames:
      #   args['epoch'] = step
      self.lr_scheduler.step(**args)

  def __call__(self, *args, **kwargs):
    self.run(*args, **kwargs)

  def stop(self, val=True):
    self._stop = val

  def checkpoint(self, name=None) -> Path:
    state = self.state_dict()
    path = self.paths.checkpoints / (
      f'{name}.th'
      if name
      else f'{timestr()}-epoch-{self.num_epochs:03d}-steps-{self.num_steps:05d}.th'
    )
    torch.save(state, str(path))
    print(f'Saved checkpoint at {path}')
    return path

  def load_checkpoint(
    self,
    path,
    metadata=True,
    map_location=None,
    strict: bool = True,
    keys=None,
  ):
    # TODO: add 'latest', 'best' support
    log.info(f'Attempting to load checkpoint {path}')
    data = torch.load(path, map_location=map_location)
    self.load_state_dict(data, metadata=metadata, strict=strict, keys=keys)

  def export(self, path=None, trace=False, meta=None, postprocess=None):
    path = path or self.paths.exports / timestr()
    export(
      model=self.model,
      preprocess=self.params.val_transform or self.params.transform,
      postprocess=postprocess,
      classes=self.classes,
      trace=trace and next(iter(self.loader)),
      path=path,
      meta=dict(
        name=self.name,
        root=str(self.paths.root),
        dataset=get_dataset_name(self.dataset),
        num_steps=self.num_steps,
        num_samples=self.num_samples,
        num_epochs=self.num_epochs,
        batch_size=self.params.batch_size,
        time_created=self.time_created,
        param_hash=hash_params(self.model),
      ),
    )

    return path

  def state_dict(self):
    data = {
      'metadata': {
        'id': self.id,
        'name': self.name,
        'num_steps': self.num_steps,
        'num_samples': self.num_samples,
        'num_epochs': self.num_epochs,
        'batch_size': self.params.batch_size,
        'time_created': self.time_created,
        'param_hash': hash_params(self.model),
      },
    }

    for k, v in self.__dict__.items():
      if hasattr(v, 'state_dict'):
        data[k] = {
          'state_dict': v.state_dict(),
          'class': fully_qualified_name(v),
        }

    return data

  def load_state_dict(
    self,
    data,
    metadata=True,
    strict: bool = True,
    keys=None,
  ):
    """
    TODO: add a way to specify which parts should be loaded (ex: model only)
    """
    skipped = set()

    from inspect import getfullargspec

    for k, v in data.items():
      if keys and k not in keys:
        continue
      if 'state_dict' in v and hasattr(self, k):
        entry = getattr(self, k)
        if 'strict' in getfullargspec(entry.load_state_dict).args:
          entry.load_state_dict(v['state_dict'], strict=strict)
        else:
          entry.load_state_dict(v['state_dict'])
        log.debug(f'loaded {k}')
      else:
        log.debug(f'skipped loading {k}')
        skipped.add(k)

    if metadata and 'metadata' in data:
      skipped.discard('metadata')
      for k, v in data['metadata'].items():
        try:
          setattr(self, k, v)
        except ValueError:
          log.warning(f'failed to set {k}')

    if skipped:
      log.warning(f'skipped {skipped} when loading checkpoint')

  def _get_env(self):
    return yann.utils.env_info()

  def update_summary(self):
    self.summary.update(
      dict(
        id=self.id,
        name=self.name,
        path=str(self.paths.root),
        num_steps=self.num_steps,
        num_samples=self.num_samples,
        num_epochs=self.num_epochs,
        batch_size=self.params.batch_size,
        device=str(self.device),
        time_created=self.time_created,
        # params={k: str(v) for k, v in self.params.items()},
      ),
    )

    if 'env' not in self.summary:
      self.summary['env'] = self._get_env()

    if self.dataset:
      if 'dataset' not in self.summary:
        self.summary['dataset'] = {}
      self.summary['dataset'].update(
        dict(
          name=get_dataset_name(self.dataset),
          size=len(self.dataset),
          num_classes=len(self.dataset.classes)
          if hasattr(self.dataset, 'classes')
          else None,
        ),
      )
    if self.model:
      if 'model' not in self.summary:
        self.summary['model'] = {}
      self.summary['model'].update(
        dict(
          name=yann.get_model_name(self.model),
          param_count=yann.param_count(self.model),
          trainable_param_count=yann.param_count(
            yann.trainable(self.model.parameters()),
          ),
        ),
      )

  def save_summary(self):
    try:
      yann.save(self.summary, self.paths.summary)
    except Exception as e:
      log.warning(
        f'Failed to save summary, most likely due to unserializable params, {e}',
      )
    return self.paths.summary

  def __str__(self):
    return f"""
id: {self.id}
name: {self.name}
root: {self.root}
batch_size: {self.params.batch_size}
device: {self.device}

PARAMS
======
{self.params}

MODEL
=====

{self.model}


DATASET
=======

{self.loader.dataset}

VALIDATION DATASET
=======

{self.val_loader.dataset if self.val_loader is not None else None}

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
      f'Trainer('
      f'\n  id={self.id},'
      f'\n  name={self.name},'
      f'\n  root={self.root},'
      f'\n  batch_size={self.params.batch_size},'
      f'\n  device={self.device}'
      '\n)'
    )

  def __getstate__(self):
    return self.__dict__

  def __setstate__(self, state):
    self.__dict__.update(state)
