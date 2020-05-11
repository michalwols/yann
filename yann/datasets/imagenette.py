from pathlib import Path
import os
import tarfile
from glob import iglob
import random
from torchvision.datasets import utils

from . import ClassificationDataset


def extract(src, dst=None):
  dst = dst or os.path.dirname(src)
  with tarfile.open(src, 'r:gz') as tar:
      tar.extractall(path=dst)


class Imagenette(ClassificationDataset):
  """
  https://github.com/fastai/imagenette
  """
  urls = {
    160: 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz',
    320: 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette-320.tgz',
    None: 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz'
  }

  @classmethod
  def get_dirname(cls, url):
    return url.split('/')[-1][:-4]

  @classmethod
  def get_filename(cls, url):
    return url.split('/')[-1]

  def __init__(self, root='./datasets/', size=None, split='train', download=True, shuffle=True):
    if size not in self.urls:
      raise ValueError(
        f"Unsupported size '{size}', "
        f"must be one of '{', '.join(self.urls.keys())}'")
    self.size = size
    self.root = Path(root).expanduser()
    self.size_root = self.root / self.get_dirname(self.urls[self.size])

    self.split = split

    if not self.size_root.exists():
      if not download:
        raise ValueError(
          f'Could not find dataset at provided root ({self.size_root}) and download=False'
        )
      self.download()

    paths = list(iglob(f'{self.size_root}/{self.split}/**/*.*', recursive=True))
    if shuffle:
      random.shuffle(paths)
    self.inputs, self.targets = list(zip(*(
      (p, p.split('/')[-2]) for p in paths
    )))

    super(Imagenette, self).__init__(classes=sorted(set(self.targets)))

  @property
  def paths(self):
    return self.inputs

  def download(self):
    self.size_root.mkdir(parents=True, exist_ok=True)
    url = self.urls[self.size]
    root = self.root
    filename = self.get_filename(url)
    utils.download_url(url, root, filename)
    print('extracting file', root / filename)
    extract(root / filename)


class Imagenette160(Imagenette):
  size = 160
  def __init__(self, **kwargs):
    if 'size' in kwargs and kwargs['size'] != self.__class__.size:
      raise ValueError('size is not a valid argument')
    super(Imagenette160, self).__init__(
      size=self.__class__.size,
      **kwargs
    )

class Imagenette320(Imagenette):
  size = 320
  def __init__(self, **kwargs):
    if 'size' in kwargs and kwargs['size'] != self.__class__.size:
      raise ValueError('size is not a valid argument')
    super(Imagenette320, self).__init__(
      size=self.__class__.size,
      **kwargs
    )

class Imagewoof(Imagenette):
  urls = {
    160: 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-160.tgz',
    320: 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-320.tgz',
    None: 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof.tgz'
  }


class Imagewoof160(Imagenette):
  size = 160
  def __init__(self, **kwargs):
    if 'size' in kwargs and kwargs['size'] != self.__class__.size:
      raise ValueError('size is not a valid argument')
    super(Imagewoof160, self).__init__(
      size=self.__class__.size,
      **kwargs
    )


class Imagewoof320(Imagenette):
  size = 320
  def __init__(self, **kwargs):
    if 'size' in kwargs and kwargs['size'] != self.__class__.size:
      raise ValueError('size is not a valid argument')
    super(Imagewoof320, self).__init__(
      size=self.__class__.size,
      **kwargs
    )