import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch import nn

from ..data.classes import smooth as label_smoothing

def _reduce(x, reduce=True, reduction=None):
  if not reduce:
    return x
  if reduction == 'mean' or reduction == 'elementwise_mean':
    return x.mean()
  if reduction == 'sum':
    return x.sum()
  if reduction is None or reduction == 'none':
    return x
  raise ValueError(f'Unsupported reduction')

def soft_target_cross_entropy(
    inputs,
    targets,
    smooth=None,
    reduce=True,
    dim=1,
    reduction='mean'):
  """"like cross_entropy but using soft targets"""
  if smooth:
    targets = label_smoothing(targets, smooth)

  vals = torch.sum(-targets * F.log_softmax(inputs, dim=dim), dim=dim)
  return _reduce(vals, reduce=reduce, reduction=reduction)



class SoftTargetCrossEntropyLoss(_Loss):
  def __init__(self, smooth=None, reduce=True, dim=1, reduction='mean'):
    super().__init__(reduce=reduce, reduction=reduction)
    self.reduce = reduce
    self.reduction = reduction
    self.smooth = smooth
    self.dim = dim

  def forward(self, inputs, targets):
    return soft_target_cross_entropy(
      inputs,
      targets,
      smooth=self.smooth,
      reduce=self.reduce,
      dim=self.dim,
      reduction=self.reduction)



def binary_focal_loss(logits, targets, gamma=2, alpha=None, pos_weight=None, reduce=True, reduction='mean'):
  """
  Binary focal loss (with sigmoids)

  References:
    - https://arxiv.org/abs/1708.02002 - Focal Loss for Dense Object Detection
  Notes:
    - define:
        (pt = p if y = 1 else 1-p), where p is the model estimate for the class (y=1)
        modulating factor:  (1 − pt)^γ
        focusing parameter: γ  (gamma)
    - One notable property of cross entropy loss is that even examples that are
      easily classified (pt ≫ .5) incur a loss with non-trivial magnitude.
      When summed over a large number of easy examples, these small loss
      values can overwhelm the rare class.
    - When an example is misclassified and pt is small,
      the modulating factor is near 1 and the loss is unaffected.
      As pt → 1, the factor goes to 0 and the loss for well-classified
      examples is down-weighted.
    - The focusing parameter γ smoothly adjusts the rate at which easy
      examples are downweighted. When γ = 0, FL is equivalent to CE,
      and as γ is increased the effect of the modulating factor is
      likewise increased (we found γ = 2 to work best in our experiments)

  FL(pt) = −αt(1 − pt)^γ log(pt).

  Args:
    logits: logits (before activation is applied)
    targets: targets of same shape as logits
    gamma: "focusing parameter", gamma >= 0
    alpha: class balance parameters (class weights)

  Returns:

  """
  # TODO try this numerically stable version https://github.com/richardaecn/class-balanced-loss/issues/1

  probs = torch.sigmoid(logits)
  bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction='none')

  pt = targets * probs + (1 - targets) * (1 - probs)
  modulate = 1 if gamma is None else (1 - pt) ** gamma

  focal_loss = modulate * bce

  if alpha is not None:
    assert 0 <= alpha <= 1
    alpha_weights = targets * alpha + (1 - targets) * (1 - alpha)
    focal_loss *= alpha_weights

  return _reduce(focal_loss, reduce=reduce, reduction=reduction)


class BinaryFocalLoss(_Loss):
  def __init__(self, gamma=2, alpha=None, pos_weight=None, reduce=True, reduction='mean'):
    super(BinaryFocalLoss, self).__init__()

    self.gamma = gamma
    self.alpha = alpha
    self.pos_weight = pos_weight
    self.reduce = reduce
    self.reduction = reduction

  def forward(self, logits, targets):
    return binary_focal_loss(
      logits,
      targets,
      gamma=self.gamma,
      alpha=self.alpha,
      pos_weight=self.pos_weight,
      reduction=self.reduction
    )

class ClassWeighted(_Loss):
  def __init__(self, loss, weights=None, reduce=True, reduction='mean'):
    super(ClassWeighted, self).__init__(reduce=reduce, reduction=reduction)
    self.loss = loss(reduction='none') if issubclass(loss, _Loss) else loss
    self.register_buffer('weights', weights)
    self.reduce = reduce
    self.reduction = reduction

  def forward(self, outputs, targets):
    loss = self.loss(outputs, targets)
    if self.weights is not None:
      loss = loss * self.weights
    return _reduce(loss, reduce=self.reduce, reduction=self.reduction)


def contrastive_loss():
  pass


def triplet_loss():
  pass




def tempered_log(x, temperature=1):
  if temperature == 1: return torch.log(x)
  return (x ** (1 - temperature) - 1) / (1 - temperature)


def tempered_exp(x, temperature=1):
  if temperature == 1: return torch.exp(x)
  return torch.relu(1 + (1 - temperature) * x) ** (1 / (1 - temperature))


def bi_tempered_logistic_loss():
  """
  https://arxiv.org/pdf/1906.03361.pdf
  Returns:
  """
  pass


def bi_tempered_binary_logistic_loss():
  pass



class WeightedLoss(_WeightedLoss):
  def __init__(self, loss, weight=1, **kwargs):
    super(WeightedLoss, self).__init__(weight=weight, **kwargs)
    self.loss = loss

  def forward(self, *input, **kwargs):
    loss = self.weight * self.loss(*input, **kwargs)
    return _reduce(loss, reduction=self.reduction, reduce=self.reduce)

class CombinedLoss(_Loss):
  def __init__(self, losses, weights, *args, **kwargs):
    super(CombinedLoss, self).__init__(*args, **kwargs)
    self.losses = nn.ModuleList(
      WeightedLoss(loss, weight, reduction='none')
      for loss, weight in zip(losses, weights)
    )

  def forward(self, *input, **kwargs):
    loss = sum((loss(*input, **kwargs) for loss in self.losses))
    return _reduce(loss, reduction=self.reduction, reduce=self.reduce)


# class MultiTaskLoss(_Loss):
#   pass