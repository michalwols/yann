from .base import FunctionCallback
from .checkpoint import Checkpoint
from .eval import MulticlassEval
from .history import History, HistoryPlotter, HistoryWriter
from .logging import Logger
from .stop import StopOnNaN
from .timing import Timing


def get_callbacks(
    interactive=True,
    plot=True,
    write=True,
    log=True,
    checkpoint=True,
    time=False,
):
  return [
    x for x in (
      History(),
      plot and HistoryPlotter(save=not interactive),
      write and HistoryWriter(),
      checkpoint and Checkpoint(),
      log and Logger(),
      time and Timing()
    ) if x]
