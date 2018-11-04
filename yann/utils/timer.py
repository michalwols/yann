from collections import OrderedDict
from contextlib import contextmanager

from datetime import datetime


class Timer:
  def __init__(self, name=None, log=False):
    self.tasks = OrderedDict()
    self.name = name
    self.log = log

  def start(self, name, **extra):
    self.tasks[name] = {
      'start': datetime.now(),
      **extra
    }

    if self.log:
      print('Started', name)

  def end(self, name, **extra):
    task = self.tasks.get(name, {})
    task['end'] = datetime.now()
    task = {**extra, **task}
    self.tasks[name] = task

    if self.log:
      print('Ended', name, ', took',
            (task['end'] - task['start']).total_seconds(), 'seconds')

  @contextmanager
  def task(self, name, **extra):
    self.start(name, **extra)
    yield self
    self.end(name, **extra)

  def summary(self):
    d = OrderedDict()
    for name, data in self.tasks.items():
      start = data.get('start')
      end = data.get('end')

      x = {**data}
      if start and end:
        x['seconds'] = (end - start).total_seconds()
      d[name] = x
    return d

  def __enter__(self):
    self.start(self.name)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.end(self.name)
