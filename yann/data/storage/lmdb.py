import lmdb

from ..serialize import serialize, deserialize


class LMDB:
  NONE = '__LMDB_NONE__'

  def __init__(self, path, **kwargs):
    self.path = path
    self.db = None
    self.open(**kwargs)

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
      return self.deserialize(t.get(self.serialize(key)))

  def __setitem__(self, key, value):
    with self.db.begin(write=True) as t:
      return t.put(*self.serialize(key, value))

  def __delitem__(self, key):
    with self.db.begin(write=True) as t:
      return t.delete(self.serialize(key))

  def __iter__(self):
    with self.db.begin(write=False) as t:
      for k, v in t.cursor():
        yield self.deserialize(k, v)

  def serialize(self, key, value=NONE):
    if value is LMDB.NONE:
      return key
    else:
      return key, value

  def deserialize(self, key, value=NONE):
    if value is LMDB.NONE:
      return key
    else:
      return key, value
