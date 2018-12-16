import base64
import os

import io
import numpy as np
import pathlib
import torch
from PIL import Image
from torchvision import transforms as tvt
from torchvision.transforms.functional import to_pil_image

from ..utils import truthy


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

  def __repr__(self):
    return (
      f"{self.__class__.__name__}(\n"
      f"  load={str(self.load)}\n"
      f"  transform={str(self.transform)}\n"
      f"  to_tensor={str(self.to_tensor)}\n"
      ")")


class ImageTransformer(Transformer):
  def __init__(
      self,
      resize=None,
      rotate=None,
      crop=None,
      warp=None,
      mirror=None,
      mean=None,
      std=None,
      color_jitter=None,
      interpolation=None,
      color_space=None,
      load=None,
      transform=None,
      to_tensor=None
  ):
    interpolation = interpolation or Image.ANTIALIAS
    self.resize = resize and tvt.Resize(resize, interpolation=interpolation)
    self.rotate = rotate and tvt.RandomRotation(rotate)
    self.crop = crop and (
      tvt.RandomResizedCrop(crop, interpolation=interpolation)
      if warp
      else tvt.CenterCrop(crop)
    )
    self.mirror = mirror and tvt.RandomHorizontalFlip(
      .5 if mirror is True else mirror)

    if color_jitter is True:
      color_jitter = (.4, .2, .1, .05)
    self.color_jitter = color_jitter and tvt.ColorJitter(*color_jitter)

    self.normalize = (mean or std) and tvt.Normalize(mean=mean, std=std)

    super().__init__(
      load=load or GetImage(color_space),
      transform=tvt.Compose(truthy([
        self.resize,
        transform,
        self.rotate,
        self.crop,
        self.mirror,
        self.color_jitter,
      ])),
      to_tensor=to_tensor or tvt.Compose(truthy([
        tvt.ToTensor(),
        self.normalize
      ]))
    )


class GetImage:
  def __init__(self, space=None):
    self.color_space = space

  def __call__(self, x):
    return get_image(x, self.color_space)


def get_image(x, space=None) -> Image.Image:
  if isinstance(x, Image.Image):
    return x.convert(space) if space else x

  if isinstance(x, np.ndarray):
    img = Image.fromarray(x)
    return img.convert(space) if space else img

  if isinstance(x, torch.Tensor):
    img = to_pil_image(x)
    return img.convert(space) if space else img

  if isinstance(x, (str, pathlib.Path)) and os.path.exists(x):
    img = Image.open(x)
    return img.convert(space) if space else img

  if isinstance(x, str):
    if x.startswith('http') or x.startswith('www.'):
      import requests
      x = requests.get(x).content
    elif x.startswith('data') and 'base64,' in x:
      # data header for base64 encoded
      x = x.split('base64,')[1]
      x = base64.b64decode(x)
    elif len(x) > 1024:
      # assume no header base 64 image
      try:
        x = base64.b64decode(x)
      except:
        pass

  if hasattr(x, 'read'):
    img = Image.open(io.BytesIO(x.read()))
    return img.convert(space) if space else img

  if isinstance(x, bytes):
    img = Image.open(io.BytesIO(x))
    return img.convert(space) if space else img
