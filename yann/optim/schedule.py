from torch.optim.lr_scheduler import LRScheduler


class WarmupStableDecayScheduler(LRScheduler):
  """
  WSD Scheduler
  """

  def __init__(self, optimizer, max_lr, warmup_steps, decay_steps):
    super().__init__()
