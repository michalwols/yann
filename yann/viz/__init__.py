from .image import show_images
from .plot import (
  plot_confusion_matrix,
  plot_cooccurrences,
  plot_line,
  plot_pred_scores,
  plot_rocs,
)


class Plotter:
  def __call__(self, *args, **kwargs):
    pass


class Shower:
  def __call__(self, x, format=None, **kwargs):
    if isinstance(x, (list, tuple)):
      if isinstance(x[0], str):
        if x[0].lower().endswith(('.jpg', '.jpeg', '.png')):
          return self.images(x, **kwargs)

      try:
        from PIL import Image

        if isinstance(x[0], Image.Image):
          return self.images(x, **kwargs)
      except:
        pass
    if isinstance(x, str):
      if x.lower().endswith(('.jpg', '.jpeg', '.png')):
        return self.images(x, **kwargs)

    try:
      from PIL import Image

      if isinstance(x, Image.Image):
        return self.images(x, **kwargs)
    except:
      pass

    import torch

    if isinstance(x, torch.Tensor):
      return self.tensor(x)

    return x

  def images(self, *args, **kwargs):
    return show_images(*args, **kwargs)

  def tensor(self, t, *args, **kwargs):
    from .html import tensor

    return tensor(t, *args, **kwargs).display()


show = Shower()
plot = Plotter()
