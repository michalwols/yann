import base64
import os

import io
import numpy as np
import pathlib
import torch
import random
from PIL import Image
from torchvision import transforms as tvt
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms

from ..utils import truthy


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




def mixup(inputs, targets, alpha=1):
  """
  Args:
    inputs: batch of inputs
    targets: hot encoded targets
  Returns:
    mixed up (inputs, targets)

  """
  shuffled_indices = torch.randperm(inputs.shape[0], device=inputs.device)
  fraction = np.random.beta(alpha, alpha)
  return (
    fraction * inputs + (1 - fraction) * inputs[shuffled_indices],
    fraction * targets + (1 - fraction) * targets[shuffled_indices]
  )



def cutout(img, percent=.3, value=0):
  pil_img = False
  if isinstance(img, Image.Image):
    img = np.array(img)
    pil_img = True
  height, width = img.shape[:2]

  mask_height = round(height * percent)
  mask_width = round(width * percent)

  start_h = random.randint(0, (height - mask_height))
  start_w = random.randint(0, (width - mask_width))

  img[start_h:start_h + mask_height, start_w:start_w + mask_width] = value
  return Image.fromarray(img) if pil_img else img



def get_imagenet_transformers(size=224, resize=256, fixres=False):
  train_transform = Transformer(
    load=GetImage('RGB'),
    transform=transforms.Compose([
#       transforms.Resize(resize, interpolation=Image.ANTIALIAS),
      transforms.RandomResizedCrop(
          size,
#           scale=(.4, 1),
#           interpolation=Image.ANTIALIAS
      ),
      transforms.ColorJitter(
          .3, .3, .3
#           brightness=.4, contrast=.2, saturation=.1, hue=.05
      ),
      transforms.RandomHorizontalFlip(),
    ]),
    to_tensor=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225])
    ])
  )

  test_transform = Transformer(
    load=train_transform.load,
    transform=transforms.Compose([
      transforms.Resize(size, interpolation=Image.ANTIALIAS),
      transforms.CenterCrop(size)
    ]),
    to_tensor=train_transform.to_tensor
  )

  return train_transform, test_transform
