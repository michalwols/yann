from yann.modules.conv.mixconv import MixConv
import torch

import pytest


def test_mixconv():
  c = MixConv(10, 20, (3,5,7))

  assert c.input_channel_counts == [4, 3, 3]
  assert c.output_channel_counts == [8, 6, 6]

  t = torch.rand((8, 10, 32, 32))

  x = c(t)

  assert x.shape == torch.Size([8, 20, 32, 32])

def test_not_depthwise():
  c = MixConv(10, 20, (3, 5), depthwise=False)
  t = torch.rand((8, 10, 32, 32))

  # NOTE: there's a bug in mkl-dnn that makes this fail
  #  when there's a filter of shape 7
  x = c(t)
  # assert False


def test_1_group():
  c = MixConv(4, 4, 3)

  assert len(c.convs) == 1

  t = torch.rand((10, 4, 32, 32))
  c(t)

def test_auto_groups():
  c = MixConv((2,2,2), (2,2,2))

  assert c.kernel_sizes == [3,5,7]

def test_variable_input_channel_counts():
  c = MixConv((16, 8, 4, 4), (16, 8, 4, 4), (3, 5, 7, 9))

  t = torch.rand((8, 32, 32, 32))

  x = c(t)

  assert x.shape

  c = MixConv((16, 8, 4, 4), (8, 4, 2, 2), (3, 5, 7, 9))

  c(t)

  c = MixConv((16, 8, 4, 4), (16, 16, 4, 4), (3, 5, 7, 9))
  c(t)

  c = MixConv((16, 8, 4, 4), (16, 16, 4, 4), (3, 5, 7))
  c(t)