from pathlib import Path

from ..data import flatten
from ..data.io import load_json
from .functional import step, train
from .trainer import Trainer


def collect_summaries(root='.', name='summary.json', pandas=True):
  s = [load_json(f) for f in Path(root).glob(f'**/*{name}')]
  if pandas:
    import pandas as pd

    return pd.DataFrame([flatten(x) for x in s])
  else:
    return s


import yann.train.track
