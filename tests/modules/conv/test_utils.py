from yann.modules.conv.utils import get_same_padding


def test_get_same_padding():
  assert get_same_padding(3) == 1
  assert get_same_padding(5) == 2
  assert get_same_padding(7) == 3

