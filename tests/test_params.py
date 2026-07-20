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


def test_hyperparams_allows_attribute_assignment():
  class Params(params.HyperParams):
    value: int = 1

  params_instance = Params()

  params_instance.value = 3

  assert params_instance.value == 3


def test_hyperparams_fork_returns_same_type_with_updates():
  class Params(params.HyperParams):
    value: int = 1
    other: str = 'x'

  params_instance = Params()

  forked = params_instance.fork(value=5)

  assert isinstance(forked, Params)
  assert forked.value == 5
  assert forked.other == 'x'
  assert params_instance.value == 1


def test_hyperparams_to_dict_and_from_dict(tmp_path):
  class Params(params.HyperParams):
    value: int = 1
    other: str = 'x'

  instance = Params(value=7)

  data = instance.to_dict()
  assert data == {'value': 7, 'other': 'x'}

  restored = Params.from_dict(data)
  assert isinstance(restored, Params)
  assert restored.value == 7
  assert restored.other == 'x'

  snapshot_path = tmp_path / 'params.json'
  params.save_params(instance, snapshot_path)
  assert snapshot_path.exists()

  helper_dict = params.to_dict(instance)
  assert helper_dict == data


def test_to_serializable_dict_handles_objects():
  class Params(params.HyperParams):
    loader = object()

  serializable = params.to_serializable_dict(Params())

  assert 'loader' in serializable
  assert isinstance(serializable['loader'], str)
