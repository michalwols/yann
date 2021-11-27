import wandb

from yann.callbacks import Callback
from yann.train import Trainer


class Wandb(Callback):
  def __init__(self, project=None, entity=None):
    self.client = wandb
    self.project = project
    self.entity = entity

    self.batch_log_freq = 10

  def on_train_start(self, trainer: Trainer = None):
    self.client.init(
      project=self.project,
      entity=self.entity,
      name=trainer.name,
      config=dict(trainer.params) if trainer.params else {}
    )

  def on_train_end(self, trainer=None):
    self.client.finish()

  def on_step_end(
    self,
    index=None,
    inputs=None,
    targets=None,
    outputs=None,
    loss=None,
    trainer=None
  ):
    if trainer.num_steps % self.batch_log_freq == 0:
      self.client.log({'train/loss': loss}, step=trainer.num_steps)
      for metric, values in trainer.history.metrics.items():
        self.client.log({f'train/{metric}': values[-1]}, step=len(values) - 1)

  def on_validation_end(self, targets=None, outputs=None, loss=None, trainer=None):
    for metric, values in trainer.history.val_metrics.items():
      self.client.log({f'validation/{metric}': values[-1]}, step=trainer.num_steps)