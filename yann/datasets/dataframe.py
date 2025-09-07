import pathlib

import pandas as pd
from torch.utils.data import Dataset

import yann
from yann.data import Classes


class DataFrame(Dataset):
  data: pd.DataFrame

  def __init__(self, source, columns=None, target_col=None):
    if isinstance(source, (str, pathlib.Path)):
      source = yann.load(source)
    self.data = source

    self.columns = columns

    self.target_col = target_col
    if target_col:
      self.classes = Classes.from_labels(self.data[target_col])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    row = self.data.iloc[index]
    if self.columns:
      return row[self.columns]
    return row
