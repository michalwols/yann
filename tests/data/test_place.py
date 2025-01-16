import torch
import pytest

from yann.data.place import Place


devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']


@pytest.mark.parametrize('device', devices)
def test_place(device):
  import torch
  tuple_batch = (torch.rand(3,3), torch.rand(3,1), 'foo')

  place = Place(('cpu', device))
  b = place(tuple_batch)
  # assert b[0].device == 'cpu'
  assert b[2] == 'foo'