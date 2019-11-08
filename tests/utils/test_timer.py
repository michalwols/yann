from yann.utils.timer import Timer


def test_timer():
  t = Timer()

  t.start('foo')

  t.end('foo')


  t.start('bar')
  t.start('bar')
  t.end('bar')
  t.end('bar')