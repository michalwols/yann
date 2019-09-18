from yann.modules.loss import binary_focal_loss
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from yann.testing import check_tensor

T = torch.tensor


def test_focal_loss():
  logits = torch.Tensor([
      [-10, 2.3],
      [-10, 5.5]
    ])

  targets = torch.Tensor([
      [1.0, 0.0],
      [.0, 1.0]
    ])

  assert torch.allclose(
    binary_focal_loss(
      logits, targets, reduction='none', gamma=0, alpha=None),
    binary_cross_entropy_with_logits(
      logits, targets, reduction='none')
  ), "focal loss with gamma == 0 and no alpha should be same as binary cross entropy loss"

  assert torch.allclose(
    binary_focal_loss(
      logits, targets, reduction='mean', gamma=0, alpha=None),
    binary_cross_entropy_with_logits(
      logits, targets, reduction='mean')
  ), "focal loss with gamma == 0 and no alpha should be same as binary cross entropy loss"

  assert not torch.allclose(
    binary_focal_loss(
      logits, targets, reduction='none', gamma=.4, alpha=None),
    binary_cross_entropy_with_logits(
      logits, targets, reduction='none')
  ), "focal loss with gamma > 0 should not be same as binary cross entropy loss"


  assert binary_focal_loss(
    T([[100.0]]),
    T([[1.0]]),
    gamma=4
  ) == binary_focal_loss(
    T([[-100.0]]),
    T([[0.0]]),
    gamma=4
  ), "loss should be symmetrical"

  assert binary_focal_loss(
    T([[100.0]]),
    T([[0.0]])
  ) == binary_focal_loss(
    T([[-100.0]]),
    T([[1.0]])
  )

  assert binary_focal_loss(
    T([[2.0]]),
    T([[1.0]]),
    gamma=1
  ) > binary_focal_loss(
    T([[2.0]]),
    T([[1.0]]),
    gamma=2
  ), "larger gamma should reduce loss for well classified examples"

  assert binary_focal_loss(
    T([[2.0]]),
    T([[1.0]]),
    gamma=1
  ) < binary_cross_entropy_with_logits(
    T([[2.0]]),
    T([[1.0]]),
  ), "focal loss should penalize well classified examples less than binary cross entropy"

  for g in range(0, 10):
    for logit in range(-1_000, 1_000, 100):
      check_tensor(
        binary_focal_loss(
          T([[float(logit)]]),
          T([[1.0]]),
          gamma=g
        ),
        anomalies=True,
        gte=0,
      )
  check_tensor(
    binary_focal_loss(
      T([[-.3]]),
      T([[.2]]),
      gamma=2
    ),
    anomalies=True,
    gte=0,
  )

def test_focal_loss_gradients():
  logits = T([[0.0], [0.0]], requires_grad=True)
  targets = T([[0.0], [0.0]])

  loss = binary_focal_loss(logits, targets)
  loss.backward()
  check_tensor(logits.grad, anomalies=True)