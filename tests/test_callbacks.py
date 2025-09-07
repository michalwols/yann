from yann import callbacks


def test():
  cbs = callbacks.get_callbacks(
    checkpoint={'freq': 5},
    plot=True,
    progress=False,
  )

  assert not any(isinstance(x, callbacks.ProgressBar) for x in cbs)
  assert any(isinstance(x, callbacks.Checkpoint) for x in cbs)
  assert any(isinstance(x, callbacks.HistoryPlotter) for x in cbs)

  for cb in cbs:
    if isinstance(cb, callbacks.Checkpoint):
      assert cb.freq == 5
