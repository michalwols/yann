from .plot import plot_line, plot_pred_scores, plot_rocs, \
  plot_confusion_matrix, plot_cooccurrences
from .image import show_images



class Plotter:
  def __call__(self, *args, **kwargs):
    pass

class Shower:
  def __call__(self, x, format=None, **kwargs):
    if isinstance(x, (list, tuple)):
      if isinstance(x[0], str):
        if x[0].lower().endswith(('.jpg', '.jpeg', '.png')):
          return self.images(x)

      try:
        from PIL import Image
        if isinstance(x[0], Image.Image):
          return self.images(x)
      except:
        pass
    if isinstance(x, str):
      if x.lower().endswith(('.jpg', '.jpeg', '.png')):
        return self.images(x)

    try:
      from PIL import Image
      if isinstance(x, Image.Image):
        return self.images(x)
    except:
      pass

    import torch
    if isinstance(x, torch.Tensor):
      return self.tensor(x)


    return x

  def images(self, images):
    return show_images(images)

  def tensor(self, t, *args, **kwargs):
    from .html import tensor
    return tensor(t, *args, **kwargs).display()

show = Shower()
plot = Plotter()