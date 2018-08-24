from matplotlib import pylab as plt
import numpy as np


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
    show=True,
    legend=True,
    name=None,
    grid=True
):

  if figsize:
    fig = plt.figure(figsize=figsize)

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

  if window != 1:
    y = [np.mean(y[i:i + window])
         for i in range(0, len(y), window)]

  if x is None:
    x = [n * window for n in range(len(y))]
  elif len(x) != len(y):
    x = [np.mean(x[i:i + window])
         for i in range(0, len(x), window)]

  plt.plot(x, y, stroke, lw=line_width, label=name and str(name))
  if grid:
    plt.grid()

  if legend:
    plt.legend(loc='best')


  if show:
    plt.show()



