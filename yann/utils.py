import numpy as np


def moving_average(data, window=10):
  cumsum = np.cumsum(np.insert(data, 0, 0))
  return (cumsum[window:] - cumsum[:-window]) / float(window)