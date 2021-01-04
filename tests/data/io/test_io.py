import yann
import torch


def test_io(tmpdir):
  x = {
    'a': 2
  }

  yann.save(x, tmpdir / 'x.json')
  y = yann.load(tmpdir / 'x.json')
  assert x == y

  yann.save(x, tmpdir / 'x.th')
  y = yann.load(tmpdir / 'x.th')
  assert x == y

  yann.save(x, tmpdir / 'x.pickle')
  y = yann.load(tmpdir / 'x.pickle')
  assert x == y