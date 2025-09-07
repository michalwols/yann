from collections import deque
from functools import partial

import numpy as np
import torch

from .utils import to_numpy


def threshold_targets(metric, threshold=0.5, **defaults):
  def m(preds, targets, **kwargs):
    return metric(preds, targets > threshold, **defaults, **kwargs)

  return m


def get_preds(scores):
  score, preds = torch.max(scores, dim=1)
  return preds


def top_k(scores, k=5, largest=True):
  return torch.topk(scores, k=k, dim=1, largest=largest, sorted=True)


def accuracy(targets: torch.Tensor, preds: torch.Tensor):
  if len(targets.shape) == 2:
    _, targets = torch.max(targets, dim=1)
  if len(preds.shape) == 2:
    _, preds = torch.max(preds, dim=1)
  return (targets == preds).sum().float() / len(preds)


def top_k_accuracy(targets, preds, k=1):
  if len(targets.shape) == 2:
    _, targets = torch.max(targets, dim=1)
  scores, preds = preds.topk(k, 1, True, True)
  preds = preds.t()
  correct = preds == targets.view(1, -1).expand_as(preds)

  return correct.sum().float() / len(targets)


def mAP(targs, preds, pos_thresh=0.5):
  preds = preds.to('cpu').numpy()
  targs = (targs.to('cpu') > pos_thresh).float().numpy()
  if np.size(preds) == 0:
    return 0
  ap = np.zeros((preds.shape[1]))
  # compute average precision for each class
  for k in range(preds.shape[1]):
    # sort scores
    scores = preds[:, k]
    targets = targs[:, k]
    # compute average precision
    ap[k] = average_precision(scores, targets)
  return 100 * ap.mean()


def average_precision(output, target):
  epsilon = 1e-8

  # sort examples
  indices = output.argsort()[::-1]
  # Computes prec@i
  total_count_ = np.cumsum(np.ones((len(output), 1)))

  target_ = target[indices]
  ind = target_ == 1
  pos_count_ = np.cumsum(ind)
  total = pos_count_[-1]
  pos_count_[np.logical_not(ind)] = 0
  pp = pos_count_ / total_count_
  precision_at_i_ = np.sum(pp)
  precision_at_i = precision_at_i_ / (total + epsilon)

  return precision_at_i


top_3_accuracy = partial(top_k_accuracy, k=3)
top_5_accuracy = partial(top_k_accuracy, k=5)
top_10_accuracy = partial(top_k_accuracy, k=10)


def precision_at_k(targets, outputs, k=5):
  scores, top_preds = top_k(outputs, k=k)
  raise NotImplementedError()


def recall_at_k(targets, outputs, k=5):
  scores, top_preds = top_k(outputs, k=k)
  raise NotImplementedError()


def hits_at_k():
  pass


def mean_reciprocal_rank():
  pass


# def average_precision(targets, preds, target_threshold=0):
#   targets, preds = to_numpy(targets), to_numpy(preds)
#   if target_threshold is not None:
#     targets = targets > target_threshold
#   return metrics.average_precision_score(targets, preds)


def label_ranking_average_precision(targets, preds, target_threshold=0):
  targets, preds = to_numpy(targets), to_numpy(preds)
  if target_threshold is not None:
    targets = targets > target_threshold

  # Inlined NumPy implementation
  n_samples, n_labels = targets.shape
  scores = np.zeros(n_samples)

  for i in range(n_samples):
    true_indices = np.flatnonzero(targets[i])
    if len(true_indices) == 0:
      continue

    pred_ranking = np.argsort(preds[i])[::-1]
    ranks = np.empty_like(pred_ranking)
    ranks[pred_ranking] = np.arange(1, n_labels + 1)  # 1-based rank

    true_ranks = ranks[true_indices]
    relevant_ranks_sorted = np.sort(true_ranks)

    precisions = np.arange(1, len(true_indices) + 1) / relevant_ranks_sorted
    scores[i] = np.mean(precisions)

  return np.mean(scores)


def coverage_error(targets, preds, target_threshold=0):
  targets, preds = to_numpy(targets), to_numpy(preds)
  if target_threshold is not None:
    targets = targets > target_threshold

  # Inlined NumPy implementation
  n_samples, n_labels = targets.shape
  max_ranks = np.zeros(n_samples)

  for i in range(n_samples):
    true_indices = np.flatnonzero(targets[i])
    if len(true_indices) == 0:
      # Assign 0 coverage error if no true labels, consistent with sklearn
      max_ranks[i] = 0
      continue

    pred_ranking = np.argsort(preds[i])[::-1]
    ranks = np.empty_like(pred_ranking)
    ranks[pred_ranking] = np.arange(1, n_labels + 1)  # 1-based rank

    true_ranks = ranks[true_indices]
    max_ranks[i] = np.max(true_ranks)

  # sklearn definition: average over samples of (max_rank - 1) / n_labels
  # Let's match that definition for consistency if users expect it
  # The definition actually seems to be just mean(max_rank) / n_labels for scikit-learn.
  # Let's stick to that.
  return np.mean(max_ranks) / n_labels if n_labels > 0 else 0


def average_precision_at_k(targets, preds, k=5):
  raise NotImplementedError()


def mean_average_precision_at_k(targets, preds, k=5):
  raise NotImplementedError()


def evaluate_multiclass(targets, outputs, preds=None, classes=None):
  preds = preds or get_preds(outputs)
  targets, outputs, preds = (
    to_numpy(targets),
    to_numpy(outputs),
    to_numpy(preds),
  )
  raise NotImplementedError()


def evaluate_multilabel(
  targets,
  outputs,
  preds=None,
  classes=None,
):
  targets, outputs, preds = (
    to_numpy(targets),
    to_numpy(outputs),
    to_numpy(preds),
  )
  raise NotImplementedError()


class Meter:
  def __init__(self):
    self.reset()

  def reset(self):
    self.max = None
    self.min = None
    self.sum = 0
    self.count = 0

  def update(self, val):
    if self.max is None or val > self.max:
      self.max = val
    if self.min is None or val < self.min:
      self.min = val

    self.sum += val
    self.count += 1

  @property
  def average(self):
    return self.sum / self.count


class WindowMeter:
  def __init__(self, length=10):
    self.length = length
    self.values = deque(maxlen=length)

  def reset(self):
    self.values = deque(maxlen=self.length)

  def update(self, val):
    self.values.append(val)

  @property
  def sum(self):
    return sum(self.values)

  @property
  def max(self):
    return max(self.values)

  @property
  def min(self):
    return min(self.values)

  @property
  def average(self):
    if not self.values:
      return None
    return sum(self.values) / len(self.values)


def exp_moving_avg(cur, prev=None, alpha=0.05, steps=None):
  """exponential moving average"""
  if prev is None:
    return cur
  avg = alpha * cur + prev * (1 - alpha)
  return avg / (1 - alpha**steps) if steps else avg


def moving_average(data, window=10):
  cumsum = np.cumsum(np.insert(data, 0, 0))
  return (cumsum[window:] - cumsum[:-window]) / float(window)
