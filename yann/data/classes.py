import numpy as np


class TargetTransformer:
  def encode(self, x, many=True):
    pass

  def decode(self, x, many=True):
    pass

  def state_dict(self):
    pass

  def load_state_dict(self, data):
    pass


class Classes(TargetTransformer):
  valid_encodings = {
    'index',
    'one_hot',
    'normalized_one_hot'
  }

  def __init__(
      self,
      classes,
      meta=None,
      default_encoding='index'
  ):
    self.classes = list(classes)
    self.indices = {c: i for i, c in enumerate(self.classes)}
    self.meta = meta

    self.dtype = 'float32'

    assert default_encoding in self.valid_encodings, \
      f'default_encoding must be one of {self.valid_encodings}, got {default_encoding}'
    self.default_encoding = default_encoding

  def __repr__(self):
    c = min(len(self.classes) // 2, 3)

    return (
      f"Classes(\n" 
      f"  count={len(self)},\n" 
      f"  default_encoding={self.default_encoding}\n"
      f"  classes=[{', '.join(self.classes[:c])}, ..., {', '.join(self.classes[-c:])}]\n"
      # f"  encoded={self.encode(self.classes[:c])}, ..., {self.encode(self.classes[-c:])}\n"
      f")"
    )

  def state_dict(self):
    return {
      'classes': self.classes,
      'meta': self.meta,
      'default_encoding': self.default_encoding
    }

  def load_state_dict(self, data):
    self.classes = data['classes']
    self.indices = {c: i for i, c in enumerate(self.classes)}
    self.meta = data['meta']
    self.default_encoding = data['default_encoding']

  def __getitem__(self, idx):
    return self.classes[idx]

  def __iter__(self):
    return iter(self.classes)

  def __contains__(self, cls):
    return cls in self.indices

  def __len__(self):
    return len(self.classes)

  def __eq__(self, other):
    return self.indices == other.indices

  def encode(self, seq, encoding=None):
    return getattr(self, (encoding or self.default_encoding) + '_encode')(seq)

  def decode(self, encoded, encoding=None):
    return getattr(self, (encoding or self.default_encoding) + '_decode')(
      encoded)

  def index_encode(self, classes):
    if isinstance(classes, (str, int)):
      return self.indices[classes]
    return [self.indices[c] for c in classes]

  def index_decode(self, indices):
    if isinstance(indices, int):
      return self[indices]
    return [self[idx] for idx in indices]

  def one_hot_encode(self, classes):
    if isinstance(classes, (str, int)):
      classes = [classes]
    y = np.zeros(len(self), dtype=self.dtype)
    y[[self.indices[c] for c in classes]] = 1
    return y

  def normalized_one_hot_encode(self, classes):
    if isinstance(classes, (str, int)):
      classes = [classes]
    y = np.zeros(len(self), dtype=self.dtype)
    y[[self.indices[c] for c in classes]] = 1
    y = np.true_divide(y, y.sum())
    return y

  def ranked_decode(self, scores):
    indices = np.argsort(scores)
    return [(self.classes[i], scores[i]) for i in indices][::-1]


def smooth(y, eps=.1, num_classes=None):
  if not num_classes:
    if len(y.shape) == 1:
      num_classes = len(y)
    else:
      num_classes = y.shape[1]
  return y * (1 - eps) + eps * (1.0 / num_classes)
