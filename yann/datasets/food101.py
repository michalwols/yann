import os.path
import random
import tarfile
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

import yann
from yann.data.classes import Classes


def extract(src, dst=None):
  dst = dst or os.path.dirname(src)
  with tarfile.open(src, 'r:gz') as tar:
    tar.extractall(path=dst)


class Food101(Dataset):
  url = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
  filename = 'food-101.tar.gz'

  splits = {
    'train',
    'test'
  }

  def __init__(
      self,
      root=None,
      split='train',
      download=True,
      shuffle=True
  ):
    assert split in self.splits
    self.root = Path(root) if root else yann.default.dataset_root(Food101)
    self.split = split

    if not self.root.exists() and download:
      self.download()

    with open(self.meta_path) as f:
      self.samples = [
        (self.get_image_path(name.strip()), name.split('/')[0])
        for name in f
      ]

    if shuffle:
      random.shuffle(self.samples)

    self.classes = Classes(sorted(set(x[1] for x in self.samples)))

  @property
  def meta_path(self):
    return f'{self.root}/meta/{self.split}.txt'

  def get_image_path(self, name):
    return f'{self.root}/images/{name.strip()}.jpg'

  def __getitem__(self, index):
    return self.samples[index]

  def __len__(self):
    return len(self.samples)

  def download(self):
    self.root.mkdir(parents=True, exist_ok=True)
    download_url(self.url, str(self.root), self.filename)
    extract(self.root / self.filename, self.root)



class Food101N(Food101):
  """
  https://kuanghuei.github.io/Food-101N/
  """
  url = 'https://iudata.blob.core.windows.net/food101/Food-101N_release.zip'
  filename = 'Food-101N_release.zip'
  pass