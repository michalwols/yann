from typing import Optional

try:
  import wandb
except:
  pass

from yann.callbacks import Callback


class Wandb(Callback):
  dist_placement = 0

  run: Optional['wandb.wandb_sdk.wandb_run.Run']

  def __init__(
      self,
      project=None,
      entity=None,
      name=None,
      watch_freq=0,
      # log_code=True,
      batch_log_freq=10,
      trackers=None,
  ):
    self.client = wandb
    self.run = None
    self.project = project
    self.entity = entity
    self.name = name
    self.batch_log_freq = batch_log_freq
    self.watch_freq = watch_freq

    self.trainer = None

    self.trackers = trackers

  def log(self, *args, **kwargs):
    self.run.log(*args, **kwargs, step=self.trainer.num_steps)

  def on_train_start(self, trainer: 'yann.train.Trainer' = None):
    self.trainer = trainer
    if self.run is None:
      self.run = self.client.init(
        project=self.project,
        entity=self.entity,
        name=self.name or trainer.name,
        config=dict(trainer.params) if trainer.params else {}
      )

    if trainer.model and self.watch_freq:
      self.run.watch(
        models=trainer.model,
        log='all',
        log_graph=True,
        log_freq=self.watch_freq
      )

    if self.trackers is None:
      self.trackers = self.get_default_trackers(trainer)

    self.run.summary.update(trainer.summary)

  def on_train_end(self, trainer=None):
    self.run.finish()

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
      self.run.log({'train/loss': loss}, step=trainer.num_steps)
      for metric, values in trainer.history.metrics.items():
        self.run.log({f'train/{metric}': values[-1]}, step=len(values) - 1)

      if self.trackers:
        for track in self.trackers:
          self.run.log(track(trainer), step=trainer.num_steps)

  def on_validation_end(self, targets=None, outputs=None, loss=None, trainer=None):
    for metric, values in trainer.history.val_metrics.items():
      self.run.log({f'validation/{metric}': values[-1]}, step=trainer.num_steps)

  def on_epoch_end(self, epoch=None, loss=None, metrics=None, trainer=None):
    self.run.summary.update(trainer.summary)
    self.run.log({'epoch': trainer.num_epochs}, step=trainer.num_steps)


  def get_default_trackers(self, trainer=None):
    if not trainer: return None
    import yann.train.track

    return [
      yann.train.track.OptimizerState()
    ]

