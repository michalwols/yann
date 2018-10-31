import shutil

import torch
from torch import nn
from torchvision import transforms

from yann.export import export, load
from yann.utils import almost_equal, equal, randimg


class Net(nn.Module):
  def __init__(self, num_classes=10, input_shape=(1, 32, 32)):
    super().__init__()
    self.num_classes = num_classes
    self.input_shape = input_shape

    self.conv = nn.Sequential(
      nn.Conv2d(1, 3, 3),
      nn.MaxPool2d(2),
      nn.ReLU(inplace=True),
      nn.Conv2d(3, 3, 3),
      nn.MaxPool2d(2),
      nn.ReLU(inplace=True),
    )

  def forward(self, input, *args, **kwargs):
    return self.conv(input)


def check_loaded_model(loaded, model):
  pass


def test_export_traced(tmpdir):
  NUM_CLASSES = 10
  model = Net(NUM_CLASSES)

  preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4,), (0.4,))
  ])

  path = tmpdir

  if path.exists():
    shutil.rmtree(str(path))

  classes = [str(n) for n in range(NUM_CLASSES)]

  export(
    path=path,
    model=model,
    trace=torch.rand(1, 1, 32, 32),
    preprocess=preprocess,
    classes=classes

  )

  expected_files = [
    'model.traced.th',
    'preprocess.pkl',
    'classes.json',
    'requirements.txt',
    'env.yml'
  ]

  for name in expected_files:
    assert (path / name).exists()

  loaded = load(path)

  assert loaded.model
  assert loaded.preprocess
  assert loaded.classes
  assert loaded.classes == classes

  input = torch.rand(1, 1, 32, 32)

  loaded.model.eval()
  model.eval()
  assert almost_equal(model(input), loaded.model(input), 1e-12)

  img = randimg(32, 32, 3)
  assert equal(preprocess(img), loaded.preprocess(img))

  assert classes == loaded.classes


def test_export_state_dict(tmpdir):
  NUM_CLASSES = 10
  model = Net(NUM_CLASSES)

  preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4,), (0.4,))
  ])

  path = tmpdir

  if path.exists():
    shutil.rmtree(str(path))

  export(
    path=path,
    model=model,
    state_dict=True,
    preprocess=preprocess,
    classes=[str(n) for n in range(NUM_CLASSES)]
  )

  expected_files = [
    'model.state_dict.th',
    'preprocess.pkl',
    'classes.json',
  ]

  for name in expected_files:
    assert (path / name).exists()

  loaded = load(path)


def test_export_pickled(tmpdir):
  NUM_CLASSES = 10
  model = Net(NUM_CLASSES)

  preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4,), (0.4,))
  ])

  path = tmpdir

  if path.exists():
    shutil.rmtree(str(path))

  export(
    path=path,
    model=model,
    state_dict=False,
    preprocess=preprocess,
    classes=[str(n) for n in range(NUM_CLASSES)]

  )

  expected_files = [
    'model.th',
    'preprocess.pkl',
    'classes.json',
  ]

  for name in expected_files:
    assert (path / name).exists()

  loaded = load(path)
