import io
import os
import pathlib

import numpy as np
from PIL import Image


def mixup(x1, x2, y1, y2, alpha=.5):
  pass


class Transformer:
  def __init__(self, load=None, transform=None, to_tensor=None):
    self.load = load
    self.transform = transform
    self.to_tensor = to_tensor

  def __call__(self, x):
    x = self.load(x)
    x = self.transform(x)
    return self.to_tensor(x)

  def prep(self, x):
    return self.transform(self.load(x))


class GetImage:
  def __init__(self, space='RGB'):
    self.color_space = space

  def __call__(self, x):
    if isinstance(x, Image.Image):
      return x.convert(self.color_space)

    if isinstance(x, (str, pathlib.Path)) and os.path.exists(x):
      with open(x, 'rb') as f:
        img = Image.open(f)
        return img.convert(self.color_space)

    if isinstance(x, np.ndarray):
      img = Image.fromarray(x)
      return img.convert(self.color_space)

    if isinstance(x, str):
      if x.startswith('http') or x.startswith('www.'):
        import requests
        x = requests.get(x).content

    if isinstance(x, bytes):
      img = Image.open(io.BytesIO(x))
      return img.convert(self.color_space)

    if hasattr(x, 'read'):
      img = Image.open(io.BytesIO(x.read()))
      return img.convert(self.color_space)
