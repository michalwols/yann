import lmdb
import pickle
import json
from contextlib import contextmanager

from ..serialize import serialize_arrow, deserialize_arrow, to_bytes, to_unicode
from ..images import image_to_bytes, image_from_bytes


class LMDB:
  def __init__(self, path, map_size=1e10, **kwargs):
    self.path = path
    self.db = None
    self.open(map_size=map_size, **kwargs)

    self._current_transaction = None

  @property
  def stats(self):
    return self.db.stat()

  def __len__(self):
    return self.stats['entries']

  def close(self):
    self.db.close()

  def open(self, **kwargs):
    self.db = lmdb.open(self.path, **kwargs)

  def __del__(self):
    self.close()
    del self.db

  def __getitem__(self, key):
    # TODO: support indexing multiple values
    if self._current_transaction:
      return self.deserialize(self._current_transaction.get(self.serialize_key(key)))
    else:
      with self.db.begin(write=False) as t:
        return self.deserialize(t.get(self.serialize_key(key)))

  def __setitem__(self, key, value):
    if self._current_transaction:
      return self._current_transaction.put(self.serialize_key(key), self.serialize(value))
    else:
      with self.db.begin(write=True) as t:
        return t.put(self.serialize_key(key), self.serialize(value))

  def __delitem__(self, key):
    if self._current_transaction:
      return self._current_transaction.delete(self.serialize_key(key))
    else:
      with self.db.begin(write=True) as t:
        return t.delete(self.serialize_key(key))

  def __iter__(self):
    if self._current_transaction:
      for k, v in self._current_transaction.cursor():
        yield self.deserialize_key(k), self.deserialize(v)
    else:
      with self.db.begin(write=False) as t:
        for k, v in t.cursor():
          yield self.deserialize_key(k), self.deserialize(v)

  def update(self, items):
    if isinstance(items, dict):
      items = items.items()
    if self._current_transaction:
      for k, v in items:
        self._current_transaction.put(self.serialize_key(k), self.serialize(v))
    else:
      with self.db.begin(write=True) as t:
        for k, v in items:
          t.put(self.serialize_key(k), self.serialize(v))

  @contextmanager
  def transaction(self, write=False, buffers=False):
    with self.db.begin(write=write, buffers=buffers) as t:
      self._current_transaction = t

      yield

      self._current_transaction = None

  @staticmethod
  def serialize_key(x):
    return x

  @staticmethod
  def deserialize_key(x):
    return x

  @staticmethod
  def serialize(x):
    return x

  @staticmethod
  def deserialize(x):
    return x


class ArrowLMDB(LMDB):
  """
  LMDB that uses arrow to serialize the values
  """

  @staticmethod
  def serialize_key(x): 
    return to_bytes(x)

  @staticmethod
  def deserialize_key(x): 
    return to_unicode(x)

  @staticmethod
  def serialize(x): 
    return serialize_arrow(x)

  @staticmethod
  def deserialize(x): 
    return deserialize_arrow(x)


class PickleLMDB(LMDB):
  @staticmethod
  def serialize_key(x):
    return to_bytes(x)

  @staticmethod
  def deserialize_key(x):
    return to_unicode(x)

  @staticmethod
  def serialize(x):
    return pickle.dumps(x, protocol=-1)

  @staticmethod
  def deserialize(x):
    return pickle.loads(x)


class ImageLMDB(LMDB):
  format = 'jpeg'

  def serialize_key(self, x):
    return to_bytes(x)

  def deserialize_key(self, x):
    return to_unicode(x)

  def serialize(self, x):
    return image_to_bytes(x, format=self.format)

  def deserialize(self, x):
    return image_from_bytes(x)


class JSONLMDB(LMDB):
  @staticmethod
  def serialize_key(x):
    return to_bytes(x)

  @staticmethod
  def deserialize_key(x):
    return to_unicode(x)

  @staticmethod
  def serialize(x):
    return to_bytes(json.dumps(x))

  @staticmethod
  def deserialize(x):
    return json.loads(to_unicode(x))