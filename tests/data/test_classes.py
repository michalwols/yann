from yann.data.classes import Classes, get_class_weights


def test_classes():
  classes = Classes.ordered(10)

  assert len(classes) == 10
  assert 1 in classes
  assert classes[1] == 1

  classes = Classes(['a', 'b', 'c'])

  assert 'a' in classes
  assert classes.indices['a'] == 0
  assert classes.index_encode('a') == 0
  assert classes.one_hot_encode('a')[0] == 1

  classes = Classes(counts={'a': 10, 'b': 20})
  assert classes.weights(normalize=False) == [30 / 10, 30 / 20]


def test_class_weights():
  counts = {0: 4, 1: 5, 2: 10}
