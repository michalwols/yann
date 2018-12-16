import os
from math import cos, pi
import numpy as np

from ..callbacks.base import Callback
from .. import set_param
from ..metrics import exp_moving_avg
from ..viz import plot_line


def cosine_anneal(min_lr, max_lr, cur_step, num_steps):
  return min_lr + (max_lr - min_lr) * (1 + cos(cur_step / num_steps * pi)) / 2


class SGDR(Callback):
  def __init__(
      self,
      optimizer=None,
      max_lr=None,
      min_lr=0,
      cycle_len=10,
      cycle_mult=1,
      lr_mult=1,
      verbose=True,
      checkpoint=False
  ):
    self.optimizer = optimizer
    self.max_lr = max_lr
    self.min_lr = min_lr
    self.base_cycle_len = cycle_len
    self.cycle_mult = cycle_mult
    self.lr_mult = lr_mult

    self.cur_cycle_len = self.base_cycle_len
    self.cur_max_lr = self.max_lr
    self.cur_min_lr = self.min_lr
    self.cur_lr = self.max_lr
    self.cur_step = 0

    self.completed_cycles = 0
    self.save_checkpoints = checkpoint
    self.checkpoints = []

    self.verbose = verbose

    if self.cur_max_lr and self.optimizer:
      self.update_lr(self.cur_lr)

  def on_train_start(self, trainer=None):
    self.trainer = trainer

  def restart(self):
    if self.save_checkpoints:
      self.checkpoints.append(self.trainer.checkpoint(
        f'cycle-{self.completed_cycles}'
        f'-epochs-{self.trainer.num_epochs}'
        f'-steps-{self.trainer.num_steps}'))

    self.cur_cycle_len *= self.cycle_mult
    self.cur_max_lr *= self.lr_mult
    self.cur_min_lr *= self.lr_mult

    self.cur_lr = self.cur_max_lr
    self.cur_step = 0
    self.update_lr(self.cur_lr)

    if self.verbose:
      print('Restarting SGDR')

  def update_lr(self, lr):
    set_param(self.optimizer, 'lr', lr)

  def on_batch_end(self, batch, inputs, targets, outputs, loss, trainer=None):
    if self.cur_step >= self.cur_cycle_len:
      self.restart()
    else:
      new_lr = cosine_anneal(self.cur_min_lr, self.cur_max_lr, self.cur_step,
                             self.cur_cycle_len)
      self.update_lr(new_lr)
      self.cur_step += 1


class LRRangeTest(Callback):
  def __init__(
      self,
      start_lr=.00001,
      end_lr=1,
      step=None,
      steps=None,
      log_freq=None,
      plot_freq=None,
      divergence_multiplier=4
  ):
    super(LRRangeTest, self).__init__()
    self.checkpoint_path = None
    self.start_lr = start_lr
    self.end_lr = end_lr

    if not step and not steps:
      raise ValueError('step or steps must be provided')

    delta = end_lr - start_lr
    self.steps = steps or (delta / step)
    self.step = step or (delta / steps)

    self.min_loss = None
    self.avg_loss = None

    self.lrs = []
    self.losses = []

    self.plot_freq = plot_freq
    self.log_freq = log_freq

    self.divergence_multiplier = divergence_multiplier

  def __repr__(self):
    return (
      f"LRRangeTest (\n"
      f"  min_lr: {self.start_lr}\n"
      f"  max_lr: {self.end_lr}\n"
      f"  step: {self.step}\n"
      f"  steps: {self.steps}\n"
      ")"
    )

  def on_train_start(self, trainer=None):
    self.checkpoint_path = trainer.checkpoint(name='lr-range-test')

    self.lrs = [self.start_lr]
    set_param(trainer.optimizer, 'lr', self.lrs[-1])

    if self.log_freq:
      print(self)

  def on_batch_end(self, batch, inputs, targets, outputs, loss, trainer=None):
    self.losses.append(loss.item())
    self.avg_loss = exp_moving_avg(self.losses[-1], self.avg_loss,
                                   steps=len(self.losses))

    if self.log_freq and len(self.lrs) % self.log_freq == 0:
      print(f"lr: {self.lrs[-1]:.5f}  loss: {self.avg_loss:.5f}")

    if self.plot_freq and len(self.lrs) % self.plot_freq == 0:
      self.plot()

    if self.min_loss is None:
      self.min_loss = self.avg_loss
    elif (self.avg_loss > self.divergence_multiplier * self.min_loss) and \
        len(self.lrs) > 50:
      trainer.stop()
      return
    elif self.avg_loss < self.min_loss:
      self.min_loss = self.avg_loss

    if len(self.lrs) >= self.steps:
      trainer.stop()
      return

    self.lrs.append(self.lrs[-1] + self.step)
    set_param(trainer.optimizer, 'lr', self.lrs[-1])

  def restore(self, trainer):
    trainer.load_checkpoint(self.checkpoint_path)
    os.remove(self.checkpoint_path)

  def on_train_end(self, trainer=None):
    self.restore(trainer)

  def on_error(self, error, trainer=None):
    self.restore(trainer)

  def plot(self, **kwargs):
    plot_line(x=self.lrs, y=self.losses, xlabel='learning rate', ylabel='loss',
              **kwargs)


class CyclicalLR(Callback):
  def __init__(self, start_lr, end_lr, steps=None, step=None):
    self.start_lr = start_lr
    self.end_lr = end_lr
    self.steps = steps

    self.step = (self.end_lr - self.start_lr) / self.steps

    self.cycle_len = 2 * self.steps
    self.cur_step = 0
    self.cur_lr = self.start_lr

  def on_train_start(self, trainer=None):
    set_param(trainer.optimizer, 'lr', self.cur_lr)

  def on_batch_end(self, batch, inputs, targets, outputs, loss, trainer=None):
    if self.cur_step % self.cycle_len // self.steps:
      self.cur_lr -= self.step
    else:
      self.cur_lr += self.step

    if self.cur_step % 100 == 0:
      print(f'lr: {self.cur_lr:.5f}')

    set_param(trainer.optimizer, 'lr', self.cur_lr)
    self.cur_step += 1
