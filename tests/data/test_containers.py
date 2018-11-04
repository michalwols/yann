from yann.data.containers import Container, Inputs


class TestContainer:
  def test_set_and_get_item(self):
    c = Container(1, 2, 3, x=5, y=6)

    assert list(c) == [1, 2, 3, 5, 6]
    assert c[-1] == 6
    assert c[:2] == [1, 2]

    assert c.args == (1, 2, 3)
    assert c.kwargs == {'x': 5, 'y': 6}

    c[3] = 9
    assert c.x == 9

    input = Inputs(image='sfsd', mask=[0, 1, 2, 3])

    image, *rest = input

    assert image == input.image == 'sfsd'
