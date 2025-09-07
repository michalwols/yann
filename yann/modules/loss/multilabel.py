import math

import torch
import torch.nn.functional as F

from yann.modules.loss import _reduce


class LargeLossNegativeRejection(torch.nn.Module):
  def __init__(
    self,
    loss=F.binary_cross_entropy_with_logits,
    threshold=None,
    percent=None,
    reduction: str = 'mean',
    pos_thresh=0.5,
  ):
    """
    Args:
      loss:
      reject: bool, if true will 0 out loss for high loss negative labels,
        if false will change the loss for high loss negatives to the positive
      reduction:
    """
    super(LargeLossNegativeRejection, self).__init__()
    self.loss = loss

    # need to disable reduction on wrapped loss
    self._loss_args = {}
    import inspect

    if hasattr(self.loss, 'reduction'):
      self.loss.reduction = 'none'
    elif 'reduction' in inspect.getfullargspec(self.loss).args:
      self._loss_args['reduction'] = 'none'

    self.threshold = threshold
    self.percent = percent
    self.reduction = reduction

    self.pos_thresh = pos_thresh

  def forward(
    self,
    preds: torch.Tensor,
    targets: torch.Tensor,
    percent=None,
    threshold=None,
  ):
    percent = percent or self.percent
    threshold = threshold or self.threshold

    losses = self.loss(preds, targets, **self._loss_args)

    unobserved_losses = losses * (targets < self.pos_thresh).bool()

    if percent is not None and percent > 0:
      unobserved_count = torch.count_nonzero(unobserved_losses)
      k = torch.ceil(unobserved_count * percent)
      largest_unobserved_losses, _ = torch.topk(
        unobserved_losses.flatten(),
        int(k),
      )
      keep_mask = (unobserved_losses < largest_unobserved_losses[-1]).float()
      losses = losses * keep_mask

    if threshold is not None:
      keep_mask = (unobserved_losses < threshold).float()
      losses = losses * keep_mask

    return _reduce(losses, reduction=self.reduction)
