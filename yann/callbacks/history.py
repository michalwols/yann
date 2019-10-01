from time import time as get_time
from pathlib import Path
import torch

from ..utils.decorators import lazy
from ..callbacks.base import Callback
from ..viz import plot_line
from .. import resolve, get_item
from ..data.metrics import MetricStore, EventStore, Event
from ..evaluation import evaluate_metrics


class History(Callback):
  def __init__(self, *metrics, **named_metrics):
    super(History, self).__init__()
    self.metric_funcs = {
      **named_metrics,
    }
    for m in metrics:
      if isinstance(m, str):
        self.metric_funcs[m] = resolve.metric(m)
      elif hasattr(m, '__name__'):
        self.metric_funcs[m.__name__] = m
      else:
        raise ValueError(f'Unknown metric {m}')

    self.metrics = MetricStore(
      self.metric_funcs.keys(),
      cast_value=get_item
    )
    self.val_metrics = MetricStore(
      self.metric_funcs.keys(),
      cast_value=get_item
    )
    self.events = EventStore()

    self.trainer = None

  def on_batch_end(self, batch, inputs, targets, outputs, loss, trainer=None):
    self.metrics.update(
      step=trainer.num_steps,
      loss=loss,
      **evaluate_metrics(targets=targets, outputs=outputs, metrics=self.metric_funcs)
    )

  def on_validation_end(self, loss=None, outputs=None, targets=None, trainer=None):
    self.val_metrics.update(
      step=trainer.num_steps,
      loss=loss,
      **evaluate_metrics(targets=targets, outputs=outputs, metrics=self.metric_funcs)
    )

  @lazy
  def plot(self):
    return HistoryPlotter(history=self)



class HistoryPlotter(Callback):
  def __init__(self, freq=500, window=50, metrics=None,
               clear=False, save=False, history: History = None):
    super().__init__()
    self.history: History = history
    self.freq = freq
    self.window = window
    self.metrics = metrics
    self.save = save

    self.figsize = (16, 16)

    self.clear = clear
    self.root = None

  def __call__(self, *args, **kwargs):
    return self.plot(*args, **kwargs)

  def plot(self, metric=None, time=False, validation=False, **kwargs):
    if self.clear:
      try:
        from IPython.display import clear_output
        clear_output(wait=True)
      except:
        pass


    if validation:
      metrics = self.history.val_metrics
    else:
      metrics = self.history.metrics

    if metric:
      names = [metric]
    else:
      names = self.metrics or metrics.keys()

    for name in names:
      plot_line(
        metrics[name],
        x=metrics.times if time else range(len(metrics)),
        xlabel='time' if time else 'step',
        ylabel=f'validation {name}' if validation else name,
        name=f'validation {name}' if validation else name,
        window=1 if validation else self.window,
        save=self.save and
             self.root / (f'validation {name}' if validation else name),
        show=not self.save,
        figsize=self.figsize,
        **kwargs
      )

  def on_train_start(self, trainer=None):
    if trainer:
      self.root = trainer.root
      self.history = self.history or trainer.history

  def on_batch_end(self, *args, trainer=None, **kwargs):
    if trainer.num_steps % self.freq == 0:
      self.plot(
        title=f'Epoch: {trainer.num_epochs} Steps: {trainer.num_steps}'
      )

  def on_validation_end(self, *args, trainer=None, **kwargs):
    self.plot(
      validation=True,
      title=f'Epoch: {trainer.num_epochs} Steps: {trainer.num_steps}'
    )


class HistoryWriter(Callback):
  def __init__(self, root=None, train=True, val=True, mode='a+',
               write_freq=1, flush_freq=500):
    self.root = root
    self.mode = mode
    self.train = train
    self.val = val

    self.flush_freq = flush_freq
    self.write_freq = write_freq

    self.header = None
    self.val_header = None

    self.train_file = None
    self.val_file = None

  @property
  def root(self):
    return self._root

  @root.setter
  def root(self, val):
    if val:
      self._root = Path(val)
      self._root.mkdir(parents=True, exist_ok=True)
    else:
      self._root = None

  def prep_files(self, root=None):
    self.root = self.root or root

    if self.train_file:
      self.train_file.close()
    if self.val_file:
      self.val_file.close()

    self.train_file = open(self.root / 'history-train.tsv', self.mode)
    self.val_file = open(self.root / 'history-val.tsv', self.mode)

  def on_train_start(self, trainer=None):
    self.prep_files(trainer.root if trainer else None)

  def on_batch_end(self, batch, inputs, targets, outputs, loss, trainer=None):
    if batch % self.write_freq:
      return

    if self.header is None:
      self.header = ['timestamp', 'epoch', 'step', *trainer.history.metrics.keys()]
      self.train_file.write('\t'.join(self.header) + '\n')

    self.train_file.write(
      f"{trainer.history.metrics.times[-1]}\t"
      f"{trainer.num_epochs}\t"
      f"{len(trainer.history.metrics.times)}\t"
      + '\t'.join((f"{trainer.history.metrics[m][-1]:.4f}"
                   for m in self.header[3:]))
      + '\n'
    )

    if batch % self.flush_freq == 0:
      self.train_file.flush()

  def on_validation_end(self, targets=None, outputs=None, loss=None,
                        trainer=None):
    if self.val_header is None:
      self.val_header = ['timestamp', 'epoch', 'step',
                         *trainer.history.val_metrics.keys()]
      self.val_file.write('\t'.join(self.header) + '\n')

    self.val_file.write(
      f"{trainer.history.val_metrics.times[-1]}\t"
      f"{trainer.num_epochs}\t"
      f"{len(trainer.history.val_metrics.times)}\t"
      + '\t'.join((f"{trainer.history.val_metrics[m][-1]:.4f}"
                   for m in self.val_header[3:]))
      + '\n'
    )

    self.val_file.flush()

  def on_train_end(self, trainer=None):
    self.train_file.close()
    self.val_file.close()

    self.train_file = None
    self.val_file = None

  def on_error(self, error, trainer=None):
    self.train_file.close()
    self.val_file.close()

    self.train_file = None
    self.val_file = None
