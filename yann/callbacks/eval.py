import sys
from sklearn.metrics import classification_report, accuracy_score
from .base import Callback

from ..metrics import get_preds


class MulticlassEval(Callback):
  def __init__(self, dest=sys.stdout):
    self.dest = dest

  def __call__(self, targets=None, outputs=None, trainer=None, **kwargs):
    if trainer and trainer.dataset and hasattr(trainer.dataset, 'classes'):
      classes = trainer.dataset.classes
    else:
      classes = None

    preds = get_preds(outputs)

    preds, targets = preds.to('cpu').numpy(), targets.to('cpu').numpy()

    print(classification_report(targets, preds, target_names=classes),
          file=self.dest)

    print('Accuracy: ', accuracy_score(targets, preds))

  def on_validation_end(self, targets=None, outputs=None, loss=None,
                        trainer=None, **kwargs):
    self(targets=targets, outputs=outputs, trainer=trainer)


class MultilabelEval(Callback):
  def __call__(self, targets=None, outputs=None, trainer=None, **kwargs):
    pass
