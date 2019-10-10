import lmdb
import pickle

from ..serialize import serialize_arrow, deserialize_arrow, to_bytes, to_unicode
from ..images import image_to_bytes, image_from_bytes


class LMDB:
  def __init__(self, path, map_size=1e10, **kwargs):
    self.path = path
    self.db = None
    self.open(map_size=map_size, **kwargs)

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
    with self.db.begin(write=False) as t:
      return self.deserialize(t.get(self.serialize_key(key)))

  def __setitem__(self, key, value):
    with self.db.begin(write=True) as t:
      return t.put(self.serialize_key(key), self.serialize(value))

  def __delitem__(self, key):
    with self.db.begin(write=True) as t:
      return t.delete(self.serialize_key(key))

  def __iter__(self):
    with self.db.begin(write=False) as t:
      for k, v in t.cursor():
        yield self.deserialize_key(k), self.deserialize(v)

  def update(self, items):
    if isinstance(items, dict):
      items = items.items()
    with self.db.begin(write=True) as t:
      for k, v in items:
        t.put(self.serialize_key(k), self.serialize(v))

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
