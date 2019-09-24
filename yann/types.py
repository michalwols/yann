import torch
import typing


Logits = torch.FloatTensor
Embedding = torch.FloatTensor

Probabilities = torch.FloatTensor
LogProbabilities = torch.FloatTensor

Batch = typing.NewType('Batch', torch.Tensor)

ClassIndices = torch.LongTensor
OneHot = torch.FloatTensor

ImageTensor = torch.ShortTensor
NormalizedImageTensor = torch.FloatTensor

RGB = ImageTensor
Grayscale = ImageTensor
