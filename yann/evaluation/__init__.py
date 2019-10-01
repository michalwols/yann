import torch

def evaluate_metrics(targets=None, outputs=None, metrics=None):
  values = {}
  with torch.no_grad():
    for name, metric in metrics.items():
      values[name] = metric(targets, outputs)
  return values