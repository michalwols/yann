import torch
from yann import to_numpy


def get_preds(scores):
  score, preds = torch.max(scores, dim=1)
  return preds


def top_k(scores, k=5, largest=True):
  return torch.topk(scores, k=k, dim=1, largest=largest)


def accuracy(targets, preds):
  if targets.shape != preds.shape:
    preds = get_preds(preds)
  return (targets == preds).sum().float() / len(preds)


def evaluate_multiclass(
    targets,
    outputs,
    preds=None,
    classes=None
):
  preds = preds or get_preds(outputs)
  targets, outputs, preds = (
    to_numpy(targets),
    to_numpy(outputs),
    to_numpy(preds)
  )






def evaluate_multilabel(
    targets,
    outputs,
    preds=None,
    classes=None,
):
  targets, outputs, preds = (
    to_numpy(targets),
    to_numpy(outputs),
    to_numpy(preds)
  )




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