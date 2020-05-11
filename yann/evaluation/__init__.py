import torch

def evaluate_metrics(targets=None, outputs=None, metrics=None):
  values = {}
  with torch.no_grad():
    for name, metric in metrics.items():
      values[name] = metric(targets, outputs)
  return values


class Evaluator:
  def update(self):
    pass

  def metrics(self):
    pass

  def report(self):
    pass

  def summary(self):
    pass

  def help(self):
    pass

  def show(self):
    pass


class MultiClassEvaluator(Evaluator):
  metrics = {}
  pass


class BinaryClassificationEvaluator(Evaluator):
  pass


class RankingEvaluator(Evaluator):
  pass


class RegressionEvaluator(Evaluator):
  pass


class RetrievalEvaluator(Evaluator):
  pass