import json
import os
import pickle as pkl
import tarfile
from collections import namedtuple
import csv
import gzip


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
    tar.extractall()


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

