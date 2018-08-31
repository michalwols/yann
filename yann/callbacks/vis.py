
from matplotlib import pylab as plt
import numpy as np


from sklearn.metrics import confusion_matrix
import itertools

import torch


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')


class VisPlotter:

  def __init__(self, classes=None, freq=500, normalize=False):
    self.freq = freq
    self.classes = classes
    self.normalize = normalize

  def on_batch_end(self, batch, inputs, targets, outputs, loss, trainer=None):
    if trainer.num_steps % self.freq == 0:
      with torch.no_grad():
        targets = targets.cpu().numpy()
        val, pred = outputs.cpu().max(1)

        cm = confusion_matrix(targets, pred)
        plt.figure(figsize=(8, 8))
        plot_confusion_matrix(cm, classes=self.classes,
                              normalize=self.normalize,
                              title='Confusion Matrix')
        plt.show()




