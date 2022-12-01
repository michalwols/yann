from yann.data.place import Place


def test_place():
  import torch
  tuple_batch = (torch.rand(3,3), torch.rand(3,1), 'foo')

  place = Place(('cpu', 'cuda'))
  b = place(tuple_batch)
  # assert b[0].device == 'cpu'
  assert b[2] == 'foo'