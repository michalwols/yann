import numpy as np
import re
import torch
from PIL import Image
import sys


def camel_to_snake(text):
  s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def abbreviate(text):
  return re.sub(r"([a-zA-Z])[a-z]*[^A-Za-z]*",r"\1", text).lower()


def get_arg_parser(x, description=None, epilog=None, parser=None, **kwargs):
    import argparse
    from ..params import Field
    parser = parser or argparse.ArgumentParser(description=description, epilog=epilog, **kwargs)
    for k, v in x.items():
        if isinstance(v, dict):
            parser.add_argument(
                f"-{abbreviate(k)}",
                f"--{camel_to_snake(k)}",
                default=v.get('default'),
                type=v.get('type'),
                action=v.get('action'),
                help=v.get('help'),
                required=v.get('required'),
                choices=v.get('choices'),
                dest=v.get('dest')
            )
        elif isinstance(v, Field):
          parser.add_argument(
            f"-{abbreviate(k)}",
            f"--{camel_to_snake(k)}",
            default=v.default,
            type=v.type,
            help=f"{v.help or k} (default: {v.default})",
            required=v.required,
            choices=getattr(v, 'choices', None),
          )
        else:
            parser.add_argument(f"-{abbreviate(k)}", f"--{camel_to_snake(k)}", default=v, type=type(v))
    return parser


def truthy(items):
  return [x for x in items if x]


class Obj(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


def almost_equal(t1, t2, prec=1e-12):
  return torch.all(torch.lt(torch.abs(torch.add(t1, -t2)), prec))


def equal(t1, t2):
  return torch.all(t1 == t2)


def randimg(*shape, dtype='uint8'):
  return Image.fromarray((np.random.rand(*shape) * 255).astype(dtype))


def progress(it, num=None):
  if not num:
    try:
      num = len(it)
    except:
      num = None

  if num:
    for n, x in enumerate(it, 1):
      sys.stdout.write(f"\r{n} / {num}")
      sys.stdout.flush()
      yield x
  else:
    for n, x in enumerate(it, 1):
      sys.stdout.write(f"\r{n}")
      sys.stdout.flush()
      yield x


def repeat(val):
  while True:
    yield val


def counter(start=0, end=None, step=1):
  current = start
  while end is None or (end and (current < end)):
    yield current
    current += step


def to_numpy(x):
  if isinstance(x, np.ndarray):
    return x
  if torch.is_tensor(x):
    return x.to('cpu').detach().numpy()
  return np.array(x)
