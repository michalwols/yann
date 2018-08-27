from .classification import Trainer


def step(model, x, y, optimizer, criterion):
  model.train()
  optimizer.zero_grad()

  pred = model(x)
  loss = criterion(pred, y)

  loss.backward()
  optimizer.step()

  return x, y, pred, loss


def train(model, batches, optimizer, criterion, device=None):
  for x, y in batches:
    if device:
      x, y = x.to(device), y.to(device)
    yield step(model, x, y, optimizer, criterion)