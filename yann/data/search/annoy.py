import logging
import os.path

from annoy import AnnoyIndex

from ..io import save_json, load_json
from .base import VectorIndex


class Annoy(VectorIndex):
  def __init__(self, path, dims=None, metric='angular', build_on_disk=True):
    self.path = path
    self.is_mutable = None
    self.is_built = None
    self.build_on_disk = build_on_disk
    self.metric = metric

    if os.path.isfile(self.path):
      logging.debug(f'Loading existing index: {self.path}')
      self.load_meta()
      assert self.dims == dims or not dims, \
        'Passed path to existing index but dims do not match'
      assert self.metric == metric or not metric, \
        'Passed path to existing index but metrics do not match'
      self.index = AnnoyIndex(self.dims, metric=self.metric)
    elif dims:
      logging.debug(
        f'Creating new index with {dims} dimensions and {self.metric} metric')
      self.dims = dims
      self.index = AnnoyIndex(self.dims, metric=self.metric)
      if build_on_disk:
        self.index.on_disk_build(self.path)
    else:
      logging.debug(f'Loading existing index: {self.path}')
      self.load_meta()
      self.index = AnnoyIndex(self.dims, metric=self.metric)

  @property
  def meta_path(self):
    return self.path + '.meta.json'

  @property
  def files(self):
    return [self.path, self.meta_path]

  def load_meta(self):
    self.__dict__.update(
      load_json(self.meta_path)
    )

  def save_meta(self):
    d = {**self.__dict__}
    d.pop('index')
    save_json(d, self.meta_path)

  def build(self, num_trees=10):
    logging.debug(f'staring to build index: {self.path}')
    self.index.build(num_trees)
    logging.debug(f'finished building index: {self.path}')
    self.is_mutable = False
    self.is_built = True
    self.save_meta()

  def save(self):
    self.index.save(self.path)
    self.is_mutable = False
    self.save_meta()

  def load(self, memory=False):
    self.index.load(self.path, prefault=memory)
    self.is_mutable = False

  def unload(self):
    self.index.unload()

  def __del__(self):
    self.unload()

  def __setitem__(self, idx, vector):
    self.index.add_item(idx, vector)

  def __getitem__(self, idx):
    return self.index.get_item_vector(idx)

  def __len__(self):
    return self.index.get_n_items()

  def add(self, vector):
    idx = len(self)
    self[idx] = vector
    return idx

  def add_bulk(self, vectors):
    start = len(self)
    for n, v in enumerate(vectors):
      self[start + n] = v
    return self

  def set_bulk(self, indices, vectors):
    for idx, vector in zip(indices, vectors):
      self[idx] = vector

  def search(self, vector, num=10, depth=None, distances=True):
    return self.index.get_nns_by_vector(vector, num, depth or -1, distances)

  def search_index(self, idx, num=10, depth=None, distances=True):
    return self.index.get_nns_by_item(idx, num, depth or -1, distances)

  def distance(self, i, j):
    return self.index.get_distance(i, j)
