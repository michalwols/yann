from collections import Counter

import numpy as np

# from enum import Enum
#
# class Encoding(Enum):
#   index = 'index'
#   one_hot = 'one_hot'
#   normalized_one_hot = 'normalized_one_hot'


class TargetTransformer:
  def encode(self, x, many=True):
    pass

  def decode(self, x, many=True):
    pass

  def state_dict(self):
    pass

  def load_state_dict(self, data):
    pass


# TODO:
# - add hierarchy support (parents, children meta) (maybe even graphs)
# - Classes(apple=meta, juice=bar) or dict
# - nn.Embeddimg with classes.embed(dims=4)
#    - looks like target embeddings were tried before: https://arxiv.org/abs/1806.10805


class Classes(TargetTransformer):
  valid_encodings = {'index', 'one_hot', 'normalized_one_hot'}

  def __init__(
    self,
    names=None,
    meta=None,
    counts=None,
    default_encoding='index',
  ):
    if names:
      self.names = list(names)
    elif meta:
      self.names = sorted(meta.keys())
    elif counts:
      self.names = sorted(counts.keys())
    else:
      raise ValueError('At least one of names, counts or meta must be defined')
    self.indices = {c: i for i, c in enumerate(self.names)}
    self.meta = meta

    self.counts = counts

    self.dtype = 'float32'

    assert default_encoding in self.valid_encodings, (
      f'default_encoding must be one of {self.valid_encodings}, got {default_encoding}'
    )
    self.default_encoding = default_encoding

  def weights(self, list=True, mode='multiclass', normalize=True):
    if self.counts:
      weights = get_class_weights(self.counts, mode=mode, normalize=normalize)
      if list:
        return [weights[n] for n in self.names]
      else:
        return weights
    else:
      raise NotImplementedError(
        'Weights can not be determined unless `counts` are set',
      )

  @classmethod
  def from_labels(cls, labels, **kwargs):
    counts = Counter()
    for l in labels:
      if isinstance(l, (str, int)):
        counts[l] += 1
      else:
        counts.update(l)
    return cls(counts=counts, **kwargs)

  @classmethod
  def ordered(cls, num, **kwargs):
    return Classes(range(num), **kwargs)

  def __repr__(self):
    c = min(len(self.names) // 2, 3)

    return (
      f'Classes(\n'
      f'  count={len(self)},\n'
      f'  default_encoding={self.default_encoding}\n'
      f'  names=[{", ".join([str(x) for x in self.names[:c]])}, ..., {", ".join([str(x) for x in self.names[-c:]])}]\n'
      # f"  encoded={self.encode(self.names[:c])}, ..., {self.encode(self.names[-c:])}\n"
      f')'
    )

  def state_dict(self):
    return {
      'names': self.names,
      'meta': self.meta,
      'default_encoding': self.default_encoding,
      'counts': self.counts,
    }

  def load_state_dict(self, data):
    if 'classes' in data:
      # fr backwards compatibility since classes was renamed to names
      self.names = data['classes']
    else:
      self.names = data['names']
    self.indices = {c: i for i, c in enumerate(self.names)}
    self.meta = data['meta']
    self.default_encoding = data['default_encoding']
    self.counts = data.get('counts')

  def __getitem__(self, idx):
    return self.names[idx]

  def __iter__(self):
    return iter(self.names)

  def __contains__(self, cls):
    return cls in self.indices

  def __len__(self):
    return len(self.names)

  def __eq__(self, other):
    return self.indices == other.indices

  def encode(self, seq, encoding=None):
    return getattr(self, (encoding or self.default_encoding) + '_encode')(seq)

  def decode(self, encoded, encoding=None):
    return getattr(self, (encoding or self.default_encoding) + '_decode')(
      encoded,
    )

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
    return [(self.names[i], scores[i]) for i in indices][::-1]

  # def truncate(self, min_count=None, topk=None, token='_OTHER_'):
  #   if self.counts is None:
  #     raise NotImplementedError("truncate not supported without counts")
  #   pass


def smooth(y, eps=0.1, num_classes=None):
  if not num_classes:
    if len(y.shape) == 1:
      num_classes = len(y)
    else:
      num_classes = y.shape[1]
  return y * (1 - eps) + eps * (1.0 / num_classes)


def get_class_weights(
  class_counts: dict,
  mode='multiclass',
  normalize=True,
  num_samples=None,
):
  """
  Args:
    class_counts: dict mapping from class to count
    mode: 'multiclass' | 'multilabel' | 'binary'
    normalize: if true will make sum of weights equal number of classes
    num_samples: count of samples in dataset, needed
  Returns:
    weights (dict): mapping from class to weight value
  """
  if mode == 'multiclass':
    num_samples = num_samples or sum(class_counts.values())
    weights = {k: num_samples / count for k, count in class_counts.items()}
    if normalize:
      scale = len(weights) / sum(weights.values())
      return {k: w * scale for k, w in weights.items()}
    else:
      return weights
  elif mode in ('binary', 'multilabel'):
    # NOTE: a bit of a hack, assuming num pos labels == num_samples
    num_samples = num_samples or sum(class_counts.values())
    weights = {
      k: (num_samples - pos_count) / pos_count for k, pos_count in class_counts.items()
    }
    if normalize:
      scale = len(weights) / sum(weights.values())
      return {k: w * scale for k, w in weights.items()}
    else:
      return weights
  else:
    raise ValueError(
      f'''Unsupported mode, got "{mode}", expected one of '''
      """multiclass, multilabel, binary""",
    )
