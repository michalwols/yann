from typing import Union, Tuple

from layers import Layer
import torch
from torch.nn import functional as F
from exceptions import ConfigurationError

from torch.nn import Conv2d as cd

from layers.core import Params


class Conv(Layer):
  def __init__(
      self,
      d: int,
      num_filters: int,
      shape: Union[int, Tuple[int]],
      bias=True,
      stride=1,
      pad=0,
      dilate=0,
      groups=1,
      init=None,
      init_bias=None,
  ):
    super(Conv, self).__init__()

    if d == 1:
      self.convolve = F.conv1d
    elif d == 2:
      self.convolve = F.conv2d
    elif d == 3:
      self.convolve = F.conv3d
    else:
      raise ConfigurationError(
        f'Conv of dimension `{d}` is not supported, '
        f'must be either 1, 2 or 3'
      )


    self.num_filters = num_filters
    self.shape = (shape,) * d if isinstance(shape, int) else d

    self.bias = bias
    self.stride = stride
    self.pad = pad
    self.dilate = dilate
    self.groups = groups

    self.init = init
    self.init_bias = init_bias

  def eval(self, tensor):
    return self.convolve(
        tensor,
        self.params.filters,
        bias=self.params.bias,
        stride=self.stride,
        padding=self.pad,
        dilation=self.dilate,
        groups=self.groups
    )


  def initialize(self, input):
    input_filters = input.shape[0]

    p = Params()

    p.filters = torch.Tensor(
      self.num_filters,
      input_filters // self.groups,
      *self.shape
    )

    if self.bias:
      p.bias = torch.Tensor(self.num_filters)
      self.init_bias(p.bias)





class Conv1D(Conv):
  def __init__(self, num_filters, shape):
    super(Conv1D, self).__init__(d=1, num_filters=num_filters, shape=shape)

  def eval(self, tensor):
    pass

class Conv2D(Conv):
  def __init__(self, num_filters, shape):
    super(Conv2D, self).__init__(d=2, num_filters=num_filters, shape=shape)


  def eval(self, tensor):

    return F.conv2d(
      tensor,
      self.filters,
      self.bias
    )


class Convolution(Conv):
  def __init__(
      self,
      d,
      num_filters,
      shape,
      activation='relu',
      dropout=None,
      batchnorm=None
  ):
    super(Convolution, self).__init__(d, num_filters, shape)

class Convolution2D(Convolution):
  pass

class SeparableConv(Conv):
  pass
