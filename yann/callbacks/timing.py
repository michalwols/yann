import time
from matplotlib import pylab as plt

from .base import Callback


class Timing(Callback):
  def __init__(self):
    super().__init__()
    self.starts = []
    self.ends = []
    self.start_time = None

  def on_train_start(self, *args, **kwargs):
    self.start_time = time.time()

  def on_batch_start(self, *args, **kwargs):
    self.starts.append(time.time())

  def on_batch_end(self, *args, **kwargs):
    self.ends.append(time.time())

  @property
  def times(self):
    return [e - s for (s, e) in zip(self.starts, self.ends)]

  @property
  def waits(self):
    return [s - e for (s, e) in zip(self.starts, [self.start_time, *self.ends])]

  def plot(self, start=0, end=None, scatter=False):
    end = end or len(self.starts)

    fig = plt.figure(figsize=(12, 4))
    plt.grid()
    if scatter:
      plt.scatter(range(start, end), list(self.times)[start:end], label='step')
      plt.scatter(range(start, end), list(self.waits)[start:end], label='prep')
    else:
      plt.bar(range(start, end), list(self.waits)[start:end], label='prep')
      plt.bar(range(start, end), list(self.times)[start:end],
              bottom=list(self.waits)[start:end], label='step')

    plt.xlabel('step')
    plt.ylabel('seconds')
    plt.title('Train Run Timings')
    plt.legend()
    plt.show()