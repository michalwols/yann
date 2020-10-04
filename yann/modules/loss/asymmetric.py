from torch import nn
import torch
from yann.typedefs import Logits, MultiLabelOneHot

from yann.modules.loss import _reduce


class AsymmetricLoss(nn.Module):
  """
  ASL - Asymmetric Loss for Multi Label Classification
  https://arxiv.org/abs/2009.14119v1

  m = probability margin
  shifted probability:
    p_m = max(p - m, 0)

  ```math
  A S L=\left\{\begin{array}{ll}
  L_{+}= & (1-p)^{\gamma_{+}} \log (p) \\
  L_{-}= & \left(p_{m}\right)^{\gamma_{-}} \log \left(1-p_{m}\right)
  \end{array}\right.
  ```

  based on official implementation:
    https://github.com/Alibaba-MIIL/ASL
  """

  def __init__(self, neg_decay=4, pos_decay=1, prob_shift=0.05, eps=1e-8, reduction='mean'):
    super(AsymmetricLoss, self).__init__()
    self.pos_decay = pos_decay
    self.neg_decay = neg_decay
    self.prob_shift = prob_shift or 0

    self.eps = eps

    self.reduction = reduction
    self.calculate_focal_loss_gradients = True


  def forward(self, inputs: Logits, targets: MultiLabelOneHot):
    pos_probs = torch.sigmoid(inputs)
    neg_probs = 1 - pos_probs

    if self.prob_shift > 0:
      # negative shifting max(p - shift, 0),
      # effectively ignores easy negative examples with small loss
      # paper claims this also helps handling mislabeled negative examples
      neg_probs = (neg_probs + self.prob_shift).clamp(max=1)

    neg_targets = (1 - targets)

    pos_losses = targets * torch.log(pos_probs.clamp(min=self.eps))
    neg_losses = neg_targets * torch.log(neg_probs.clamp(min=self.eps))
    losses = pos_losses + neg_losses  # standard cross entropy

    if self.pos_decay or self.neg_decay:
      with torch.set_grad_enabled(self.calculate_focal_loss_gradients):
        pts = pos_probs * targets + neg_probs * neg_targets
        decays = self.pos_decay * targets + self.neg_decay * neg_targets
        weights = torch.pow(1 - pts, decays)
      losses *= weights

    return _reduce(-losses, reduction=self.reduction)


class AsymmetricLossOptimized(AsymmetricLoss):
  def forward(self, inputs: Logits, targets: MultiLabelOneHot):
    self.targets = targets
    self.neg_targets = (1 - self.targets)

    self.pos_probs = torch.sigmoid(inputs)
    self.neg_probs = 1 - self.pos_probs

    if self.prob_shift > 0:
      self.neg_probs = self.neg_probs.add_(self.prob_shift).clamp_(max=1)

    self.losses = self.targets * torch.log(self.pos_probs)
    self.losses.add_(self.neg_targets * torch.log(self.neg_probs))

    if self.pos_decay or self.neg_decay:
      with torch.set_grad_enabled(self.calculate_focal_loss_gradients):
        self.pos_probs.mul_(self.targets)
        self.neg_probs.mul_(self.neg_targets)
        weights = torch.pow(
          1 - self.pos_probs - self.neg_probs,
          self.pos_decay * self.targets + self.neg_decay * self.neg_targets
        )
      self.losses *= weights

    return _reduce(-self.losses, reduction=self.reduction)