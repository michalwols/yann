from .base import FunctionCallback, Callback
from .checkpoint import Checkpoint
from .eval import MulticlassEval
from .history import History, HistoryPlotter, HistoryWriter
from .logging import Logger
from .stop import StopOnNaN
from .timing import Timing
from .progbar import ProgressBar
from .wandb import Wandb
# from .ema import EMA
# from .swa import SWA
from .lr import LRRangeTest

from .callbacks import Callbacks

from yann.utils import is_notebook

def _maybe_init(value, cls, **kwargs):
  if value is None or value is False:
    return None
  if isinstance(value, dict):
    return cls(**{**value, **kwargs})
  return cls(**kwargs)


def get_callbacks(
    interactive=None,
    plot=True,
    write=True,
    log=True,
    checkpoint=True,
    time=False,
    progress=True,
    tensorboard=True,
):

  if interactive is None:
    interactive = is_notebook()

  if tensorboard:
    from .tensorboard import Tensorboard
    tb = _maybe_init(tensorboard, Tensorboard)
  else:
    tb = None
  return [
    x for x in (
      # History(),
      _maybe_init(progress, ProgressBar, notebook=interactive),
      _maybe_init(plot, HistoryPlotter, save=not interactive),
      _maybe_init(write, HistoryWriter),
      _maybe_init(checkpoint, Checkpoint),
      _maybe_init(log, Logger),
      _maybe_init(time, Timing),
      tb
    ) if x]
