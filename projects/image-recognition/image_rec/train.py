import yann
from yann.train import Trainer
from yann.transforms import ImageTransforms


class Params(Trainer.Params):
  pass


def get_trainer(params: Params):
  transform = yann.params.apply(ImageTransforms, params)
