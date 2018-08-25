import torch
from yann import to_numpy

def get_preds(scores):
  score, preds = torch.max(scores, dim=1)
  return preds


def top_k(scores, k=5, largest=True):
  return torch.topk(scores, k=k, dim=1, largest=largest)


def accuracy(targets, preds):
  return (targets == preds).sum() / len(preds)





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
