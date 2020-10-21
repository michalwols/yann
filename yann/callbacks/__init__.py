from .base import FunctionCallback, Callback
from .checkpoint import Checkpoint
from .eval import MulticlassEval
from .history import History, HistoryPlotter, HistoryWriter
from .logging import Logger
from .stop import StopOnNaN
from .timing import Timing
from .progbar import ProgressBar

def get_callbacks(
    interactive=True,
    plot=True,
    write=True,
    log=True,
    checkpoint=True,
    time=False,
    progress=True,
    tensorboard=True,
):

  if tensorboard:
    from .tensorboard import Tensorboard
    tb = Tensorboard()
  else:
    tb = None
  return [
    x for x in (
      # History(),
      progress and ProgressBar(notebook=interactive),
      plot and HistoryPlotter(save=not interactive),
      write and HistoryWriter(),
      checkpoint and Checkpoint(),
      log and Logger(),
      time and Timing(),
      tb
    ) if x]
