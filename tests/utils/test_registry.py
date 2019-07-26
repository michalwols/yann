import pytest
from torch.optim import SGD
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets import MNIST, FashionMNIST

from yann.config.registry import Registry, RegistryError


def test_registry():
  r = Registry()

  assert len(r) == 0, "Registry didn't start out empty"
  r.register(MNIST, name='mnist')
  assert len(r) == 1

  assert 'mnist' in r._records
  assert r._records['mnist'].x is MNIST

  r2 = Registry(types=(Dataset,))

  r2.register(MNIST)

  with pytest.raises(RegistryError):
    r2.register(SGD)


def test_registry_nesting():
  r = Registry()

  assert isinstance(r.dataset, Registry)

  r.dataset.register(MNIST)
  r.dataset.fashion.register(FashionMNIST)

  assert len(r.dataset) == 2
  assert len(r.dataset.fashion) == 1


def test():
  registry = Registry()
  yann = lambda: None
  yann.registry = registry
  yann.register = registry
  yann.resolve = registry.resolve

  yann.register(MNIST, 'mnist')

  assert yann.registry.resolve('mnist', instance=False) is MNIST

  yann.register.dataset(MNIST)

  ds = list(yann.registry.dataset.values())

  yann.register.optimizers(
    SGD,
    init=lambda SGD, *args, parameters=None, **kwargs: SGD(parameters, )
  )


def test_indexing():
  registry = Registry()
  yann = lambda: None
  yann.registry = registry
  yann.register = registry
  yann.resolve = registry.resolve

  yann.registry.dataset.index(datasets, types=(Dataset,))
  assert yann.resolve.dataset('MNIST', instance=False) is datasets.MNIST

  assert yann.registry.has(datasets.MNIST)
  assert 'MNIST' in yann.registry
  assert 'MNIST' in yann.registry.dataset
  assert 'MNIST' not in yann.registry.loss

  class CustomDataset(Dataset):
    def __init__(self, required_arg):
      self.required_arg = required_arg

  yann.register.dataset(CustomDataset)

  dset = CustomDataset('value')
  assert yann.resolve.dataset(dset, instance=True) is dset

  @yann.register.dataset
  class Foobar(Dataset):
    pass

  assert yann.resolve.dataset('Foobar', instance=False) is Foobar

  @yann.register('ReLU')
  def relu(x):
    return max(0, x)

  assert yann.resolve('ReLU') is relu
  assert yann.resolve('ReLU')(2) == 2
  assert yann.resolve('ReLU')(-1) == 0

  assert yann.resolve(relu)(-1) == 0


def test_yann_registry():
  import yann

  assert len(yann.registry.dataset)
  assert len(yann.registry.loss)
  assert len(yann.registry.optimizer)

  yann.resolve('MNIST', required=True)
