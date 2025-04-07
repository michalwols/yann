from collections import defaultdict
import os.path
from pathlib import Path
import datetime
import yann

import torch
from ...utils import fully_qulified_name, timestr
from ...utils.ids import memorable_id

"""
TODO:
  - pickle
  - parquet
  - numpy
  - csv
  - json lines
  - hdf5
  - tfrecords
  - lmdb
"""


class BatchWriter:
  def __init__(self, path, encoders=None, names=None, meta=None):
    self.path = path

    if isinstance(encoders, (list, tuple)):
      if not names:
        raise ValueError('Names must be provided if encoders are a tuple')
      if len(encoders) != len(names):
        raise ValueError('names and encoders must be the same length if provided as tuples')
      encoders = dict(zip(names, encoders))

    self.encoders = encoders or {}
    self.names = names
    self.writer = None
    self.buffers = defaultdict(list)

    self.meta = meta or {}
    self.meta['encoders'] = {
      k: {
        'path': fully_qulified_name(v),
        'name': getattr(v, '__name__', None)
      } for k, v in self.encoders.items()
    }
    self.meta['time_created'] = timestr()
    self.meta['write_id'] = memorable_id()
    self.meta['path'] = str(self.path)
    self.save_meta()

  def encode_batch(self, *args, **kwargs):
    if args:
      items = zip(self.names, args)
    else:
      items = kwargs.items()

    data = {}
    for k, v in items:
      if self.encoders and k in self.encoders:
        v = self.encoders[k][v]
      elif torch.is_tensor(v):
        v = v.detach().cpu()

      data[k] = v
    return data

  def batch(self, *args, **kwargs):
    data = self.encode_batch(*args, **kwargs)
    for k, v in data.items():
      self.buffers[k].append(v)

  def through(self, batches):
    for b in batches:
      if isinstance(b, (tuple, list)):
        self.batch(*b)
      else:
        self.batch(**b)
      yield b

  def all(self, batches):
    for b in batches:
      if isinstance(b, (tuple, list)):
        self.batch(*b)
      else:
        self.batch(**b)

  def collate(self, buffers):
    return buffers

  def flush(self):
    self._write()
    self._wipe_buffers()

  @property
  def meta_path(self) -> Path:
    return Path(self.path).parent / f"writer-meta.json"

  def save_meta(self):
    yann.save(self.meta, self.meta_path)

  def close(self):
    self.flush()
    if self.writer and hasattr(self.writer, 'close'):
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self):
    self.close()

  def _wipe_buffers(self):
    for k in self.buffers:
      self.buffers[k] = []

  def _write(self):
    Path(self.path).parent.mkdir(parents=True, exist_ok=True)
    collated = self.collate(self.buffers)
    self._save(dict(collated), self.path)

  def _save(self, data, path):
    yann.save(data, path)

  def _num_buffered_batches(self):
    return len(next(iter(self.buffers.values())))


class BatchStreamWriter(BatchWriter):
  pass

class PartitionedBatchWriter(BatchWriter):
  def __init__(self, path, batches_per_file=256, encoders=None, names=None, meta=None):
    super().__init__(path, encoders=encoders, names=names, meta=meta)

    self.part = 0
    self.batches_per_file = batches_per_file

  def batch(self, *args, **kwargs):
    super().batch(*args, **kwargs)
    if self._num_buffered_batches() >= self.batches_per_file:
      self.flush()

  def get_part_path(self, part):
    if callable(self.path):
      return self.path(part=part, batches=self.buffers)
    elif '{' in self.path and '}' in self.path:
      return self.path.format(
        part=part,
        time=datetime.datetime.utcnow()
      )
    else:
      name, ext = os.path.splitext(self.path)
      return f"{name}-{part}{ext}"

  def _write(self):
    path = self.get_part_path(self.part)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    collated = self.collate(self.buffers)
    self._save(dict(collated), path)



class BatchReader:
  def __init__(self, path):
    pass

  def batches(self):
    pass

  def samples(self):
    pass

  def __iter__(self):
    return self.batches()


def writer() -> BatchWriter:
  raise NotImplementedError()


def reader() -> BatchReader:
  raise NotImplementedError()