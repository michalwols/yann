import torch

import time
from yann.callbacks.base import Callback
from yann.viz import plot_line



class History(Callback):
  def __init__(self, **metrics):
    super(History, self).__init__()
    self.metric_funcs = metrics
    self.metrics = {m: [] for m in metrics}
    self.metrics['loss'] = []
    self.times = []


    self.val_metrics = {m: [] for m in metrics}
    self.val_metrics['loss'] = []
    self.val_steps = []
    self.val_times = []

  def on_batch_end(self, batch, inputs, targets, outputs, loss, trainer=None):
    self.times.append(time.time())
    self.metrics['loss'].append(loss.item())

    with torch.no_grad():
      for name, metric in self.metric_funcs.items():
        self.metrics[name].append(
          metric(targets, outputs)
        )

  def on_validation_end(self, loss=None, outputs=None, targets=None,
    trainer=None):
    self.val_times.append(time.time())
    self.val_steps.append(trainer.num_steps)
    self.val_metrics['loss'].append(loss.item())


class HistoryPlotter(Callback):
  def __init__(self, history, freq=500, window=50, metrics=None, clear=False):
    super().__init__()
    self.history = history
    self.freq = freq
    self.window = window

    self.metrics = metrics

    self.clear = clear

  def plot(self, metric=None, time=False, **kwargs):
    if not metric:
      if self.clear:
        try:
          from IPython.display import clear_output
          clear_output(wait=True)
        except:
          pass
      metrics = self.metrics or self.history.metrics.keys()
      for m in metrics:
        plot_line(
          self.history.metrics[m],
          x=self.history.times if time else None,
          xlabel='step',
          ylabel=m,
          window=self.window,
          **kwargs
        )
    else:
      plot_line(
        self.history.metrics[metric],
        x=self.history.times if time else None,
        xlabel='step',
        ylabel=metric,
        window=self.window,
        **kwargs
      )

  def on_batch_end(self, *args, trainer=None, **kwargs):
    if trainer.num_steps % self.freq == 0:
      self.plot()




      