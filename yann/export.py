import os
import shutil
import subprocess
from contextlib import suppress
from pathlib import Path

import torch

from .data import Classes
from .data.io import (
  load_pickle, load_json, save_json, save_pickle, tar_dir,
  untar
)


# TODO: add way to pass validation data to check model outputs when loaded again
def export(
    model=None,
    preprocess=None,
    postprocess=None,
    predict=None,
    classes=None,
    trace=False,
    state_dict=False,
    path=None,
    # validation=None,
    meta=None,
    tar=False,
    **kwargs
):
  os.makedirs(path, exist_ok=True)
  if os.listdir(path):
    raise ValueError(
      f'Failed to export because {path} already exists and is not empty')

  path = Path(path)

  if model:
    if trace is not False and trace is not None:
      from torch import jit
      traced = jit.trace(model, trace)
      traced.save(os.path.join(path, 'model.traced.th'))
    else:
      if not state_dict:
        torch.save(
          model,
          os.path.join(path, 'model.th')
        )
      else:
        torch.save(
          model.state_dict(),
          os.path.join(path, 'model.state_dict.th')
        )
  if preprocess:
    save_pickle(preprocess, path / 'preprocess.pkl')

  if postprocess:
    save_pickle(postprocess, path / 'postprocess.pkl')

  if meta:
    save_json(meta, path / 'meta.json')

  if classes:
    save_json(
      classes.state_dict() if isinstance(classes, Classes) else classes,
      path / 'classes.json'
    )

  if kwargs:
    save_pickle(kwargs, path / 'kwargs.pkl')

  if predict:
    save_pickle(predict, path / 'predict.pkl')

  subprocess.call(
    ['conda', 'env', 'export', '-f', os.path.join(path, 'env.yml')]
  )

  subprocess.call(
    ['pip', 'freeze'], stdout=open(os.path.join(path, 'requirements.txt'), 'w')
  )

  if tar:
    tar_dir(path)
    shutil.rmtree(str(path))


class Predictor:
  def __init__(self):
    self.model = None
    self.classes = None
    self.meta = {}
    self.kwargs = {}

    self.model_state_dict = None

    self.postprocess = None
    self.predict = None
    self.postprocess = None

  def predict(self):
    pass


def load(path, eval=True):
  path = Path(path)
  p = Predictor()

  # TODO: read from tarfile directly instead
  if str(path).endswith('.tar.gz'):
    untar(path)
    path = Path(str(path).rstrip('.tar.gz'))

  p.model = None
  if (path / 'model.th').exists():
    p.model = torch.load(str(path / 'model.th'))
  elif (path / 'model.traced.th').exists():
    from torch import jit
    p.model = jit.load(str(path / 'model.traced.th'))

  if p.model and eval:
    p.model.eval()

  with suppress(FileNotFoundError):
    p.model_state_dict = torch.load(str(path / 'model.state_dict.th'))

  with suppress(FileNotFoundError):
    p.classes = load_json(path / 'classes.json')

  with suppress(FileNotFoundError):
    p.preprocess = load_pickle(path / 'preprocess.pkl')

  with suppress(FileNotFoundError):
    p.postprocess = load_pickle(path / 'postprocess.pkl')

  with suppress(FileNotFoundError):
    p.predict = load_pickle(path / 'predict.pkl')

  with suppress(FileNotFoundError):
    p.meta = load_json(path / 'meta.json')

  with suppress(FileNotFoundError):
    p.kwargs = load_pickle(path / 'kwargs.pkl')

  return p
