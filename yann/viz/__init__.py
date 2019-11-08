from .plot import plot_line, plot_pred_scores, plot_rocs, \
  plot_confusion_matrix, plot_cooccurrences
from .image import show_images



class Plotter:
  def __call__(self, *args, **kwargs):
    pass

class Shower:
  def __call__(self, *args, **kwargs):
    pass

show = Shower()
plot = Plotter()