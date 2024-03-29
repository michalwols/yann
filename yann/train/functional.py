

def step(model, inputs, targets, optimizer, loss, callback: 'yann.callbacks.callback.Callback' = None):
  model.train()
  optimizer.zero_grad()

  pred = model(inputs)
  loss = loss(pred, targets)

  loss.backward()
  optimizer.step()

  return inputs, targets, pred, loss


def train(model, batches, optimizer, loss, device=None, step=step, callback: 'yann.callbacks.callback.Callback'=None):
  for inputs, targets in batches:
    if device:
      inputs, targets = inputs.to(device), targets.to(device)
    yield step(model, inputs, targets, optimizer, loss)