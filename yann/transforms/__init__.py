import base64
import io
import os
import pathlib
import random
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision import transforms as tvt
from torchvision.transforms.functional import to_pil_image

try:
  from timm.data.mixup import rand_bbox
except ImportError:
  HAS_TIMM = False
else:
  HAS_TIMM = True

from ..utils import truthy


class Transform(Protocol):
  def __call__(self):
    pass


class FittableTransform(Transform):
  def fit(self, x: Any): ...

  def transform(self, x: Any) -> Any: ...

  def fit_transform(self, x: Any) -> Any: ...


class Transforms:
  def __init__(self, load=None, transform=None, to_tensor=None):
    self.load = load
    self.transform = transform
    self.to_tensor = to_tensor

  def __call__(self, x):
    x = self.load(x)
    x = self.transform(x)
    return self.to_tensor(x)

  def trace(self, x):
    loaded = self.load(x)
    yield loaded
    transformed = self.transform(loaded)
    yield transformed
    tensor = self.to_tensor(transformed)
    yield tensor

  def prep(self, x):
    return self.transform(self.load(x))

  def __repr__(self):
    return (
      f'{self.__class__.__name__}(\n'
      f'  load={str(self.load)}\n'
      f'  transform={str(self.transform)}\n'
      f'  to_tensor={str(self.to_tensor)}\n'
      ')'
    )


# for backwards compatibility after rename, to avoid confusion with "Transformers"
Transformer = Transforms


class ImageTransforms(Transforms):
  def __init__(
    self,
    load=None,
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
    transform=None,
    to_tensor=None,
    autoaugment=None,
    randaugment=None,
    trivialaugment=None,
    erase=None,
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
      0.5 if mirror is True else mirror,
    )

    if color_jitter is True:
      color_jitter = (0.4, 0.2, 0.1, 0.05)
    self.color_jitter = color_jitter and tvt.ColorJitter(*color_jitter)

    self.normalize = (mean or std) and tvt.Normalize(mean=mean, std=std)

    self.erase = erase and (
      tvt.RandomErasing(erase)
      if isinstance(erase, float)
      else tvt.RandomErasing(*erase)
    )

    self.autoagument = autoaugment and tvt.AutoAugment()
    self.randaugment = randaugment and tvt.RandAugment()
    self.trivialaugment = trivialaugment and tvt.TrivialAugmentWide()

    super().__init__(
      load=load or GetImage(color_space),
      transform=tvt.Compose(
        truthy(
          [
            self.resize,
            transform,
            self.autoagument,
            self.randaugment,
            self.trivialaugment,
            self.rotate,
            self.crop,
            self.mirror,
            self.color_jitter,
          ],
        ),
      ),
      to_tensor=to_tensor
      or tvt.Compose(truthy([tvt.ToTensor(), self.normalize, self.erase])),
    )

  def state_dict(self):
    pass

  def load_state_dict(self):
    pass


# for backwards compatibility after rename, to avoid confusion with "Transformers"
ImageTransformer = ImageTransforms


class DictTransforms:
  def __init__(self, **transforms):
    self.transforms = transforms

  def __call__(self, data: dict):
    return {
      k: (self.transforms[k](v) if k in self.transforms else v) for k, v in data.items()
    }


class BatchTransforms:
  def __init__(self, transform):
    self.transform = transform

  def __call__(self, items):
    return [self.transform(x) for x in items]


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
    fraction * targets + (1 - fraction) * targets[shuffled_indices],
  )


class Mixup:
  def __init__(self, alpha=1):
    self.alpha = alpha

  def __call__(self, inputs, targets):
    return mixup(inputs=inputs, targets=targets, alpha=self.alpha)


def cutout(img, percent=0.3, value=0):
  pil_img = False
  if isinstance(img, Image.Image):
    img = np.array(img)
    pil_img = True
  height, width = img.shape[:2]

  mask_height = round(height * percent)
  mask_width = round(width * percent)

  start_h = random.randint(0, (height - mask_height))
  start_w = random.randint(0, (width - mask_width))

  img[start_h : start_h + mask_height, start_w : start_w + mask_width] = value
  return Image.fromarray(img) if pil_img else img


if HAS_TIMM:

  def cutmix(inputs, targets, beta):
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(inputs.size()[0]).cuda()
    target_a = targets
    target_b = targets[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[
      rand_index,
      :,
      bbx1:bbx2,
      bby1:bby2,
    ]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
    return lam * target_a + (1 - lam) * target_b


def get_imagenet_transforms(
  size=224,
  crop_scale=(0.5, 1.2),
  val_size=None,
  resize=256,
  fixres=False,
  trivial=False,
):
  augment = transforms.Compose(
    [
      transforms.RandomResizedCrop(
        size,
        scale=crop_scale,
      ),
      transforms.TrivialAugmentWide(),
      # transforms.ColorJitter(
      #     .3, .3, .3
      # ),
      transforms.RandomHorizontalFlip(),
    ],
  )

  train_transform = Transforms(
    load=GetImage('RGB'),
    transform=augment,
    to_tensor=transforms.Compose(
      [
        transforms.ToTensor(),
        # transforms.RandomErasing(),
        transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
        ),
      ],
    ),
  )

  test_transform = Transforms(
    load=train_transform.load,
    transform=transforms.Compose(
      [
        transforms.Resize(val_size or size, interpolation=Image.ANTIALIAS),
        transforms.CenterCrop(val_size or size),
      ],
    ),
    to_tensor=transforms.Compose(
      [
        transforms.ToTensor(),
        transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
        ),
      ],
    ),
  )

  return train_transform, test_transform
