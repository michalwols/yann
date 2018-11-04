import json
import os
import pickle as pkl
import re
import tarfile

import numpy as np
import torch
from PIL import Image


def camel_to_snake(text):
  s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def truthy(items):
  return [x for x in items if x]


class Obj(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


def save_pickle(obj, path, mode='wb'):
  with open(str(path), mode) as f:
    pkl.dump(obj, f)


def load_pickle(path, mode='rb'):
  with open(str(path), mode) as f:
    return pkl.load(f)


def save_json(obj, path, mode='w'):
  with open(str(path), mode) as f:
    json.dump(obj, f)


def load_json(path, mode='r'):
  with open(str(path), mode) as f:
    return json.load(f)


def tar_dir(path, dest=None):
  path = str(path)
  dest = str(dest or path)

  if not dest.endswith('.tar.gz'):
    dest = os.path.splitext(dest)[0] + '.tar.gz'

  with tarfile.open(dest, 'w:gz') as tar:
    tar.add(path)


def untar(path):
  with tarfile.open(path) as tar:
    tar.extractall()


def almost_equal(t1, t2, prec=1e-12):
  return torch.all(torch.lt(torch.abs(torch.add(t1, -t2)), prec))


def equal(t1, t2):
  return torch.all(t1 == t2)


def randimg(*shape):
  return Image.fromarray((np.random.rand(*shape) * 255).astype('uint8'))
