
from pathlib import Path
from ..data.io import load_json
from ..data import flatten
from .supervised import Trainer
from ..callbacks import Callback


def step(model, inputs, targets, optimizer, loss, callback: Callback = None):
  model.train()
  optimizer.zero_grad()

  pred = model(inputs)
  loss = loss(pred, targets)

  loss.backward()
  optimizer.step()

  return inputs, targets, pred, loss


def train(model, batches, optimizer, loss, device=None, step=step, callback: Callback=None):
  for inputs, targets in batches:
    if device:
      inputs, targets = inputs.to(device), targets.to(device)
    yield step(model, inputs, targets, optimizer, loss)


def collect_summaries(root='.', name='summary.json', pandas=True):
  s = [load_json(f) for f in Path(root).glob(f'**/*{name}')]
  if pandas:
    import pandas as pd
    return pd.DataFrame([flatten(x) for x in s])
  else:
    return s