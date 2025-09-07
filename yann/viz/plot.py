import base64
import datetime
import io
import itertools
import pathlib
from collections import OrderedDict
from urllib.parse import quote

import numpy as np

from .. import to_numpy
from ..metrics import moving_average


def plot_line(
  y,
  x=None,
  figsize=None,
  window=1,
  xlim=None,
  ylim=None,
  line_width=2,
  stroke='-',
  title=None,
  xlabel=None,
  ylabel=None,
  xscale=None,
  yscale=None,
  show=True,
  save=False,
  legend=True,
  name=None,
  grid=True,
):
  from matplotlib import pylab as plt

  fig = figsize and plt.figure(figsize=figsize)

  if xlim:
    plt.xlim(xlim)
  if ylim:
    plt.ylim(ylim)

  if xlabel:
    plt.xlabel(xlabel)
  if ylabel:
    plt.ylabel(ylabel)
  if title:
    plt.title(title)

  if yscale:
    plt.yscale(yscale)

  if xscale:
    plt.xscale(xscale)

  if window != 1:
    y = moving_average(y, window=window)

  if x is None:
    x = [n * window for n in range(len(y))]
  elif len(x) != len(y):
    x = moving_average(x, window=window)

  plt.plot(x, y, stroke, lw=line_width, label=name and str(name))
  if grid:
    plt.grid(True)

  if legend:
    plt.legend(loc='best')

  if save and show:
    raise ValueError("Can't show and save at the same time")
  if show:
    plt.show()
  if save:
    plt.gcf().savefig(
      save
      if isinstance(save, (str, pathlib.Path))
      else f'{title or ylabel or datetime.datetime.utcnow().strftime("%y-%m-%dT%H%M%S")}.jpg',
    )
    plt.gcf().clear()
    if fig:
      plt.close(fig)

  return plt.gcf()


def plot_pred_scores(
  preds,
  targets,
  classes=None,
  logscale=True,
  figsize=(12, 6),
  show=True,
  save=False,
  title=None,
):
  import seaborn as sns
  from matplotlib import pylab as plt

  preds, targets = to_numpy(preds), to_numpy(targets)

  if title:
    plt.title(title)

  if classes:
    classes = (
      classes.items()
      if isinstance(classes, dict)
      else ((c, n) for n, c in enumerate(classes))
    )
  else:
    classes = ((n, n) for n in range(preds.shape[1]))

  for cls, idx in classes:
    f, ax = plt.subplots(figsize=figsize)
    if logscale:
      ax.set(yscale='log')

    if len(targets.shape) == 1:
      sns.distplot(
        preds[targets != idx, idx],
        bins=50,
        kde=False,
        rug=False,
        hist_kws={'range': [0, 1]},
        ax=ax,
        color='red',
        label='Negative',
      )
      sns.distplot(
        preds[targets == idx, idx],
        bins=50,
        kde=False,
        rug=False,
        hist_kws={'range': [0, 1]},
        ax=ax,
        color='blue',
        label='Positive',
      )
    else:
      sns.distplot(
        preds[targets[:, idx] == 0, idx],
        bins=50,
        kde=False,
        rug=False,
        hist_kws={'range': [0, 1]},
        ax=ax,
        color='red',
        label='Negative',
      )
      sns.distplot(
        preds[targets[:, idx] == 1, idx],
        bins=50,
        kde=False,
        rug=False,
        hist_kws={'range': [0, 1]},
        ax=ax,
        color='blue',
        label='Positive',
      )
    ax.set_title(cls)
    plt.xlabel('Score')
    plt.ylabel('Sample Count')
    plt.legend(loc='best')

  if show:
    plt.show()
  if save:
    plt.savefig(
      save
      if isinstance(save, (str, pathlib.Path))
      else f'{title or datetime.datetime.utcnow().strftime("%y-%m-%dT%H%M%S")}.jpg',
    )
    plt.gcf().clear()


def plot_rocs(
  preds,
  targets,
  classes=None,
  figsize=(12, 12),
  show=True,
  save=False,
  title=None,
):
  from matplotlib import pylab as plt
  from sklearn.metrics import auc, roc_curve

  preds, targets = to_numpy(preds), to_numpy(targets)

  fig = plt.figure(figsize=figsize)
  if title:
    plt.title(title)

  if classes:
    classes = (
      classes.items()
      if isinstance(classes, dict)
      else ((c, n) for n, c in enumerate(classes))
    )
  else:
    classes = ((n, n) for n in range(preds.shape[1]))

  plt.xlim([0.0, 1.02])
  plt.ylim([0.0, 1.02])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.plot([0, 1], [0, 1], 'k--', lw=2)

  vals = {}
  for cls, idx in classes:
    if len(targets.shape) == 1:
      fpr, tpr, thresholds = roc_curve(targets, preds[:, idx])
    else:
      fpr, tpr, thresholds = roc_curve(targets[:, idx], preds[:, idx])
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC = %0.3f)' % area)

    vals[cls] = (area, fpr, tpr, thresholds)
  plt.legend(loc='best')

  if show:
    plt.show()
  if save:
    plt.savefig(
      save
      if isinstance(save, (str, pathlib.Path))
      else f'{title or datetime.datetime.utcnow().strftime("%y-%m-%dT%H%M%S")}.jpg',
    )
    plt.gcf().clear()

  return vals


def plot_cooccurrences(counts, classes, figsize=(14, 11)):
  import pandas as pd
  import seaborn as sns
  from matplotlib import pylab as plt

  mask = np.ones_like(counts)
  mask[np.tril_indices_from(mask)] = False

  df_cm = pd.DataFrame(counts, index=list(classes), columns=list(classes))
  plt.figure(figsize=figsize)
  plot = sns.heatmap(
    df_cm,
    robust=True,
    annot=True,
    fmt='d',
    cmap='YlGnBu',
    mask=mask,
  )
  plot.set_title('Class Co-occurrence')


def sorted_indices(matrix, desc=False):
  inds = np.dstack(np.unravel_index(np.argsort(matrix.ravel()), matrix.shape))
  return (np.fliplr(inds) if desc else inds)[0]


def truncate_confusion_matrix(matrix, thresh=0.2, top=None, symmetric=True):
  inds = sorted_indices(matrix, desc=True)
  inds = [(r, c) for r, c in inds if r != c]

  if top:
    inds = inds[:top]

  if thresh:
    inds = [(r, c) for r, c in inds if matrix[r, c] >= thresh]

  if symmetric:
    rows = cols = sorted(set(y for x in inds for y in x))
  else:
    rows, cols = zip(*inds)
    rows, cols = sorted(set(rows)), sorted(set(cols))

  m = matrix[rows, :][:, cols]
  return m, rows, cols


def plot_confusion_matrix(
  preds,
  targets,
  classes,
  figsize=(16, 16),
  thresh=None,
  top=None,
  normalize=False,
  symmetric=True,
):
  from matplotlib import pylab as plt
  from sklearn.metrics import confusion_matrix

  preds, targets = to_numpy(preds), to_numpy(targets)

  cm = confusion_matrix(targets, preds)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  if thresh or top:
    cm, rows, cols = truncate_confusion_matrix(
      cm,
      thresh=thresh,
      top=top,
      symmetric=symmetric,
    )
  else:
    rows, cols = list(range(len(classes))), list(range(len(classes)))

  if not len(cm):
    return cm

  fig = plt.figure(figsize=figsize)
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title('Confusion matrix')
  plt.colorbar()
  plt.xticks(np.arange(len(cols)), [classes[i] for i in cols], rotation=75)
  plt.yticks(np.arange(len(rows)), [classes[i] for i in rows])

  fmt = '.2f' if normalize else 'd'
  th = cm.max() / 2.0
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
      j,
      i,
      format(cm[i, j], fmt),
      horizontalalignment='center',
      color='white' if cm[i, j] > th else 'black',
    )

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

  return cm


def plot_timeline(tasks, figsize=None):
  import matplotlib.dates as mdates
  from matplotlib import pylab as plt
  from matplotlib.collections import PolyCollection

  inds = OrderedDict((c, n) for n, c in enumerate(set(t.name for t in tasks)))
  color_map = {c: f'C{n}' for c, n in inds.items()}

  verts = []
  colors = []
  for task in tasks:
    v = [
      (mdates.date2num(task.start_time), inds[task.name] - 0.4),
      (mdates.date2num(task.start_time), inds[task.name] + 0.4),
      (mdates.date2num(task.end_time), inds[task.name] + 0.4),
      (mdates.date2num(task.end_time), inds[task.name] - 0.4),
      (mdates.date2num(task.start_time), inds[task.name] - 0.4),
    ]
    verts.append(v)
    colors.append(color_map[task.name])

  bars = PolyCollection(
    verts,
    facecolors=colors,
  )

  fig, ax = plt.subplots(figsize=figsize)
  ax.add_collection(bars)
  ax.autoscale()
  loc = mdates.SecondLocator(list(range(0, 60, 10)))
  ax.xaxis.set_major_locator(loc)
  ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))

  ax.set_yticks(list(inds.values()))
  ax.set_yticklabels(list(inds.keys()))
  plt.show()


def figure_to_base64(fig, format='png', data_encoded=True, close=False):
  from matplotlib import pylab as plt

  buf = io.BytesIO()
  fig.savefig(buf, format=format)
  buf.seek(0)
  string = base64.b64encode(buf.read())

  if close:
    plt.close(fig)

  if data_encoded:
    return f'data:image/{format};base64,{quote(string)}'

  return string
