import torch

from yann.rl import (
  clipped_policy_loss,
  correctness_gate,
  discounted_returns,
  group_advantages,
  masked_mean,
  runtime_reward,
  token_logprobs,
)


def test_masked_and_policy_functions():
  values = torch.tensor([[1.0, 2.0, 100.0]])
  mask = torch.tensor([[1.0, 1.0, 0.0]])
  assert masked_mean(values, mask).item() == 1.5

  logits = torch.randn(2, 3, 5)
  tokens = torch.randint(0, 5, (2, 3))
  logprobs = token_logprobs(logits, tokens)
  assert logprobs.shape == tokens.shape

  advantages = torch.ones_like(logprobs)
  loss = clipped_policy_loss(
    logprobs, logprobs.detach(), advantages, mask=torch.ones_like(logprobs)
  )
  assert torch.isfinite(loss)


def test_rewards_and_advantages():
  rewards = runtime_reward(
    torch.tensor([10.0, 10.0]),
    torch.tensor([5.0, 20.0]),
  )
  gated = correctness_gate(rewards, torch.tensor([True, False]))
  assert gated[0] > 0
  assert gated[1] == -1

  advantages = group_advantages(
    torch.tensor([1.0, 3.0, 10.0, 14.0]),
    torch.tensor([0, 0, 1, 1]),
  )
  assert torch.allclose(advantages[:2].mean(), torch.tensor(0.0))
  assert torch.allclose(advantages[2:].mean(), torch.tensor(0.0))

  returns = discounted_returns(torch.tensor([[1.0, 1.0, 1.0]]), gamma=1.0)
  assert torch.equal(returns, torch.tensor([[3.0, 2.0, 1.0]]))
