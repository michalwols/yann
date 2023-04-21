import yann.utils
from .base import Callback


class ProgressBar(Callback):
  def __init__(self, length=None, samples=True, notebook=None):
    """
    Args:
      length: number of steps per epoch, will infer from dataset or loader if not provided
      samples: count samples, if False will track batches/steps
      notebook: use notebook progress bar
    """
    self.notebook = notebook if notebook is not None else yann.utils.is_notebook()

    self.length = length
    self.samples = samples

  def on_epoch_start(self, epoch=None, trainer=None):
    if self.notebook:
      try:
        from tqdm.notebook import tqdm
      except:
        from tqdm import tqdm
    else:
      from tqdm import tqdm


    if trainer:
      try:
        total = (len(trainer.dataset) if self.samples else len(trainer.loader))
      except:
        total = None
    else:
      total = None

    self.bar = tqdm(
      desc=f"Epoch {epoch}",
      total=self.length or total,
      unit='samples' if self.samples else 'batches'
    )

  def on_epoch_end(
    self, epoch=None, loss=None, metrics=None, trainer=None
  ):
    self.bar.close()

  def on_step_end(
    self,
    index=None,
    inputs=None,
    targets=None,
    outputs=None,
    loss=None,
    trainer=None
  ):
    if self.samples:
      self.bar.update(len(inputs))
    else:
      self.bar.update(1)
