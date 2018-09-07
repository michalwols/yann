import sys
import logging

from .base import Callback


class Logger(Callback):
  def __init__(self, batch_freq=128, dest=sys.stdout):
    super(Logger, self).__init__()

    self.dest = dest

    self.batch_freq = batch_freq
    self.batch_string = None
    self.logged_batch_shapes = False

  def log(self, *args, sep='\t', **kwargs):
    if kwargs:
      print(
        *args,
        sep.join(f'{k}: {v}' for k, v in kwargs.items()),
        file=self.dest)
    else:
      print(*args, file=self.dest)

  def on_train_start(self, trainer=None):
    self.log('Starting training\n', trainer)

  def on_batch_end(self, batch, inputs, targets, outputs, loss, trainer=None):
    if batch % self.batch_freq == 0:
      if not self.logged_batch_shapes:
        try:
          self.log("\nBatch inputs shape:", tuple(inputs.size()),
                   "\nBatch targets shape:", tuple(targets.size()),
                   "\nBatch outputs shape:", tuple(outputs.size()), '\n')
        except:
          pass
        self.logged_batch_shapes = True

      if self.batch_string:
        self.log(self.batch_string.format(
          batch=batch, **({m: v[-1] for m, v in
                           trainer.history.metrics.items()})))
      else:
        self.log(
          batch=f'{batch:>8}',
          **({m: f'{v[-1]:.4f}' for m, v in trainer.history.metrics.items()}))

  def on_epoch_start(self, epoch, trainer=None):
    self.log('\nStarting epoch', epoch)

    self.log(f'''
OPTIMIZER
=========

{trainer.optimizer}


PROGRESS
========
epochs: {trainer.num_epochs}
steps: {trainer.num_steps}
samples: {trainer.num_samples}\n''')

  def on_validation_start(self, trainer=None):
    self.log('\nStarting Validation')

  def on_validation_end(self, loss=None, outputs=None, targets=None,
    trainer=None):
    self.log('\nCompleted Validation')
    self.log(
      epoch=trainer.num_epochs,
      steps=trainer.history.val_steps[-1],
      **{m: f'{vals[-1]:.4f}' for m, vals in trainer.history.val_metrics.items()}
    )

  def on_epoch_end(self, epoch, loss=None, metrics=None, trainer=None):
    self.log('Completed epoch', epoch,)

  def on_train_end(self, trainer=None):
    self.log('Completed training run. \n\n')


