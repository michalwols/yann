import sys
import logging

from .base import Callback


class Logger(Callback):
  def __init__(self, batch_freq=128, dest=sys.stdout):
    super(Logger, self).__init__()

    self.dest = dest

    self.batch_freq = batch_freq
    self.batch_string = 'batch: {batch:>8}\tloss: {loss:.6f}'
    self.logged_batch_shapes = False

  def log(self, *args):
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

      self.log(self.batch_string.format(batch=batch, loss=loss and loss.item()))

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
    self.log('\nCompleted Validation, loss:', loss and loss.item())

  def on_epoch_end(self, epoch, loss=None, metrics=None, trainer=None):
    self.log('Completed epoch', epoch,)

  def on_train_end(self, trainer=None):
    self.log('Completed training run. \n\n')


