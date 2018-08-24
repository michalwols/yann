
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np


TENSOR_TYPES = (list, np.ndarray, Tensor)

def resolve(x, lookups=None):
  pass


def get_var(x, grads=True, volatile=False):
  if isinstance(x, Variable): return x
  if isinstance(x, Tensor):
    return Variable(x, requires_grad=grads, volatile=volatile)
  return Variable(Tensor(x), requires_grad=grads, volatile=volatile)


def cast_to_vars():
  pass

def get_param(x, grads=True):
  if isinstance(x, Parameter): return x
  if isinstance(x, Tensor):
    return Parameter(x, requires_grad=grads)
  return Parameter(Tensor(x), requires_grad=grads)


class lazy:
  __slots__ = 'method', 'name'
  def __init__(self, method):
    self.method = method
    self.name = method.__name__
    
  def __get__(self, obj, cls):
    if obj:
      val = self.method(obj)
      setattr(obj, self.name, val)
      return val


def observe(func):
  """Decorator that allows you to observe inputs and outputs of a function"""
  pass