from collections import abc


class Place:
  """
  Helper for calling `.to(...)` on batch entries.

  ex:
    place = Place(device='cuda', memory_format=torch.channels_last)
    place(torch.rand((3,3))

    place = Place(('cuda', 'cpu'))
    place((t1, t2))

    place = Place({'inputs': 'cuda', 'targets': 'cpu'})
    place(dict(inputs=t1, targets=t2, keys=[1,2,3])

    place = Place({'inputs': dict(device='cuda', memory_format=torch.channels_last), 'targets': 'cpu'})
    place(dict(inputs=t1, targets=t2, keys=[1,2,3])
  """

  def __init__(self, placements=None, **kwargs):
    if isinstance(placements, abc.Sequence):
      self.placements = dict(enumerate(placements))
    else:
      self.placements = placements or {}

    # support single unnamed argument format, ex Place(('cuda:0', 'cpu'))
    for k, v in self.placements.items():
      if not isinstance(v, dict):
        self.placements[k] = dict(device=v)

    self.kwargs = kwargs

  def __call__(self, batch):
    if isinstance(batch, abc.Sequence):
      return batch.__class__(
        x.to(**self.placements[n]) if n in self.placements else x
        for n, x in enumerate(batch)
      )
    elif isinstance(batch, abc.Mapping):
      return batch.__class__(
        {
          k: x.to(**self.placements[n]) if n in self.placements else x
          for k, x in batch.items()
        },
      )
    elif self.placements is None and hasattr(batch, 'to'):
      return batch.to(**self.kwargs)
    else:
      # TODO: support dataclasses
      raise ValueError('Batch must be a collection or mappable type')
