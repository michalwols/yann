from yann import params


def test():
  class Params(params.HyperParams):
    a = 4
    b = 'b'

  p = Params()
  assert p.a == 4
  assert p.b == 'b'

  assert len(p) == 2

  p = Params(a=3)
  assert p.a == 3
  assert p.b == 'b'

  assert p['a', 'b'] == (3, 'b')

  for k in p:
    assert k in ('a', 'b')

  assert 'a' in p
  assert 'x' not in p


def test_serialization(tmpdir):
  class Params(params.HyperParams):
    a = 4
    b = 'b'

  p = Params()

  p.save(tmpdir / 'params.json')
  p2 = p.load(tmpdir / 'params.json')
  assert (tmpdir / 'params.json').exists()
  assert p == p2

  p.save(tmpdir / 'params.yaml')
  p2 = p.load(tmpdir / 'params.yaml')
  assert (tmpdir / 'params.yaml').exists()
  assert p == p2

  p.save(tmpdir / 'params.pkl')
  p2 = p.load(tmpdir / 'params.pkl')
  assert (tmpdir / 'params.pkl').exists()
  assert p == p2
