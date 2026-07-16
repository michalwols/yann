import torch

from yann.losses import dpo_loss, direct_opd_loss, masked_kl, policy_delta


def test_distillation_and_dpo_losses_are_finite():
  student = torch.randn(2, 4, 8, requires_grad=True)
  teacher = torch.randn(2, 4, 8)
  mask = torch.ones(2, 4)
  loss = masked_kl(student, teacher, mask)
  assert torch.isfinite(loss)
  loss.backward()

  chosen = torch.tensor([2.0, 1.0])
  rejected = torch.tensor([1.0, 0.5])
  value = dpo_loss(chosen, rejected, chosen - 0.2, rejected)
  assert torch.isfinite(value)


def test_policy_delta_loss():
  before = torch.randn(2, 3, 7)
  after = before + torch.randn_like(before) * 0.1
  anchor = torch.randn(2, 3, 7)
  student = anchor.clone().requires_grad_()
  delta = policy_delta(before, after, clip=2.0)
  loss = direct_opd_loss(student, anchor, delta, torch.ones(2, 3))
  assert torch.isfinite(loss)
  loss.backward()
