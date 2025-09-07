import torch


class SelectiveBackprop(torch.nn.Module):
  def __init__(self, model, loss, k=None, percent=None, min=None):
    super(SelectiveBackprop, self).__init__()
    self.model = model
    self.loss = loss

    self.k = k
    self.min = min
    # self.percent = percent

  def forward(self, inputs, targets):
    if self.training:
      indices = None
      with torch.inference_mode():
        outputs = self.model(inputs)
        losses = self.loss(outputs, targets, reduction='none')

        if self.k:
          _, indices = torch.topk(losses, k=self.k)
        elif self.min:
          indices = losses >= min
        # elif self.percent:

      inputs, targets = inputs[indices], targets[indices]
      outputs = self.model(inputs)
      loss = self.loss(outputs, targets)
      return loss
    else:
      return self.model(inputs)
