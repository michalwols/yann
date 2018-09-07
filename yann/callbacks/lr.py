from yann.callbacks.base import Callback
from yann import set_param
from math import cos, pi


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

    self.verbose = verbose

    if self.cur_max_lr and self.optimizer:
      self.update_lr(self.cur_lr)

  def restart(self):
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
