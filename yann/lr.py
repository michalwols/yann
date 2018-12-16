import os

from . import set_param
from .train import train, Trainer



def lr_range_test(trainer: Trainer,  min_lr=.00001, max_lr=1, steps=None,
                  step=None, log_freq=None, restore=True):
  # assert max_lr > min_lr
  if restore:
    checkpoint_path = trainer.checkpoint(name='lr-range-test')
  else:
    checkpoint_path = None

  steps = steps or len(trainer.loader)
  step = step or ((max_lr - min_lr) / steps)

  set_param(trainer.optimizer, 'lr', min_lr)
  try:
    cur_lr = min_lr
    cur_step = 0

    while cur_step < steps:
      for x, y, pred, loss in train(trainer.model, trainer.loader,
                                    trainer.optimizer, trainer.loss,
                                    trainer.device):
        yield (cur_lr, loss)

        if log_freq and cur_step % log_freq == 0:
          print(cur_lr, loss.item())

        cur_lr += step
        cur_step += 1
        set_param(trainer.optimizer, 'lr', cur_lr)



  finally:
    if restore:
      print('loading checkpoint')
      trainer.load_checkpoint(checkpoint_path)
      os.remove(checkpoint_path)






