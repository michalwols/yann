from collections import OrderedDict, defaultdict
from contextlib import contextmanager

from datetime import datetime
import torch.cuda

from ..viz.plot import plot_timeline


class Task:
  __slots__ = ('name', 'start_time', 'end_time', 'meta', 'sync')

  def __init__(self, name=None, start=None, end=None, meta=None, sync=False):
    self.name = name
    self.start_time = start
    self.end_time = end
    self.meta = meta or {}
    self.sync = sync

  def start(self, time=None, meta=None, sync=None):
    if meta:
      self.meta.update(meta)
    sync = sync if sync is not None else self.sync
    if sync and torch.cuda.is_available():
      torch.cuda.synchronize()

    self.start_time = time or datetime.now()

  def end(self, time=None, meta=None, sync=None):
    sync = sync if sync is not None else self.sync
    if sync and torch.cuda.is_available():
      torch.cuda.synchronize()

    self.end_time = time or datetime.now()

    if meta:
      self.meta.update(meta)

  @classmethod
  def begin(cls, name=None, meta=None, sync=None):
    t = cls(name=name, meta=meta, sync=sync)
    t.start()
    return t

  @property
  def seconds(self):
    if self.start_time is None or self.end_time is None:
      return None
    return (self.end_time - self.start_time).total_seconds()

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.end()


class Timer:
  def __init__(self, name=None, log=False):
    self.tasks = []
    self.name = name
    self.log = log

    self.active_tasks = {}

  def start(self, name, sync=True, **meta):
    task = self.task(name, sync=sync, **meta)
    if self.log: print('Started', name)

    if task in self.active_tasks:
      raise ValueError(f'Nesting tasks is not allowed, "{name}" was already started and not finished')
    self.active_tasks[name] = task

  def end(self, name, sync=True, **meta):
    task = self.active_tasks.pop(name)
    if not task:
      raise ValueError(f"{name} is not an active task so can't be ended")
    task.end(sync=sync, meta=meta)

    if self.log:
      print('Ended', task.name, ', took', task.seconds, 'seconds')

  def task(self, name, sync=True, **meta):
    task = Task.begin(name=name, meta=meta, sync=sync)
    self.tasks.append(task)
    return task

  def __enter__(self):
    self.start(self.name)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.end(self.name)

  def plot(self):
    plot_timeline(self.tasks)