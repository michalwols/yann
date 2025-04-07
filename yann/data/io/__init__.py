import json
import os
import pickle as pkl
import tarfile
from collections import namedtuple
import csv
import gzip
from pathlib import Path
from typing import Union

import torch


class Loader:
  """
    gs://bucket/file.th
    ./foo/**/*.jpg
  """

  def __call__(self, path, format=None, deserialize=None, filesystem=None, **kwargs):
    path = Path(path)
    format = format or path.suffix[1:]
    if hasattr(self, format):
      return getattr(self, format)(str(path), **kwargs)
    raise ValueError(f'File format not supported ({format})')

  def th(self, path, **kwargs):
    return torch.load(path, **kwargs)

  def json(self, path, **kwargs):
    return load_json(path, **kwargs)

  def pickle(self, path, **kwargs):
    return load_pickle(path, **kwargs)

  def parquet(self, path, **kwargs):
    import pandas as pd
    return pd.read_parquet(path, **kwargs)

  def csv(self, path, **kwargs):
    import pandas as pd
    return pd.read_csv(path, **kwargs)

  def tsv(self, path, **kwargs):
    import pandas as pd
    return pd.read_csv(path, **kwargs)

  def yaml(self, path, **kwargs):
    import yaml
    with open(path, 'r') as f:
      return yaml.load(f, yaml.SafeLoader)

  def image(self, path, **kwargs):
    import PIL.Image
    return PIL.Image.open(path)

  png = image
  jpeg = image
  jpg = image
  yml = yaml
  pkl = pickle
  pt = th
  pth = th


load = Loader()


def to_pyarrow_table(x):
  import pyarrow as pa
  try:
    import pandas as pd
    if isinstance(x, pd.DataFrame):
      x = pa.Table.from_pandas(x)
  except ImportError:
    pass

  if not isinstance(x, pa.Table):
    raise ValueError(f'unsupported type {type(x)}')

  return x

class Saver:
  def __call__(
    self, x, path, format=None, serialize=None, filesystem=None, **kwargs
  ):
    path = Path(path)
    format = format or path.suffix[1:]
    if hasattr(self, format):
      return getattr(self, format)(x, path, **kwargs)
    raise ValueError(f'File format not supported ({format})')

  def txt(self, x, path):
    with open(path, 'w') as f:
      f.write(x)

  def th(self, x, path, **kwargs):
    return torch.save(x, path, **kwargs)

  def json(self, x, path, **kwargs):
    return save_json(x, path, **kwargs)

  def yaml(self, x, path, **kwargs):
    import yaml
    with open(path, 'w') as f:
      yaml.dump(x, f, sort_keys=False)

  def csv(self, x, path, **kwargs):
    import pyarrow.csv as csv
    x = to_pyarrow_table(x)
    csv.write_csv(x, path, **kwargs)

  def parquet(
      self, x: Union['pandas.Dataframe', 'pyarrow.Table'],
      path,
      **kwargs
  ):
    import pyarrow.parquet as pq
    import pyarrow as pa

    x = to_pyarrow_table(x)

    if isinstance(x, pa.Table):
      pq.write_table(x, path, **kwargs)
    else:
      raise ValueError(f'Unsupported type {type(x)} expected pandas.Dataframe or pyarrow.Table')

  def pickle(self, x, path, **kwargs):
    return save_pickle(x, path, **kwargs)

  pkl = pickle
  yml = yaml
  pt = th
  pth = th

save = Saver()


def save_pickle(obj, path, mode='wb'):
  with open(str(path), mode) as f:
    pkl.dump(obj, f)


def load_pickle(path, mode='rb'):
  with open(str(path), mode) as f:
    return pkl.load(f)


def save_json(obj, path, mode='w'):
  with open(str(path), mode) as f:
    json.dump(obj, f)
  return path


def load_json(path, mode='r'):
  with open(str(path), mode) as f:
    return json.load(f)


def tar_dir(path, dest=None, zip=True):
  path = str(path)
  dest = str(dest or path)

  ext, mode = ('.tar.gz', 'w:gz') if zip else ('.tar', 'w')

  if not dest.endswith(ext):
    dest = os.path.splitext(dest)[0] + ext

  with tarfile.open(dest, mode) as tar:
    tar.add(path)


def lines(path, mode='r'):
  with open(path, mode=mode) as f:
    yield from f


def write_lines(items, path, mode='w'):
  with open(path, mode=mode) as f:
    f.writelines(items)


def untar(path):
  with tarfile.open(path) as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner) 
        
    
    safe_extract(tar)


def unzip(zip, dest):
  import zipfile
  with zipfile.ZipFile(zip, 'r') as f:
      f.extractall(dest)


def iter_csv(path, header=True, tuples=True, sep=',', quote='"', **kwargs):
  with open(path) as f:
    reader = csv.reader(f, delimiter=sep, quotechar=quote, **kwargs)
    if header:
      if tuples:
        reader = iter(reader)
        h = next(reader)
        Row = namedtuple('Row', h)

        for r in reader:
          yield Row(*r)
      else:
        reader = iter(reader)
        h = next(reader)
        for r in reader:
          yield dict(zip(h, r))
    else:
      yield from reader


def write_csv(data, path, header=None):
  with gzip.open(path, 'wt') if path.endswith('.gz') else open(path, 'w') as f:
    writer = csv.writer(f)
    if header:
      writer.writerow(header)
    for row in data:
      writer.writerow(row)
