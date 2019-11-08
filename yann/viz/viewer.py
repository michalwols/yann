class Viewer:
  def __init__(self, data):
    self.data = data
    self.renderers = {}

  def show(self):
    pass

  def __getitem__(self):
    pass


class PandasViewer(Viewer):
  pass


class DatasetViewer(Viewer):
  pass