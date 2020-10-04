import torch
import typing


Logits = torch.FloatTensor
Embedding = torch.FloatTensor
OneHot = torch.Tensor

Probabilities = torch.FloatTensor
LogProbabilities = torch.FloatTensor

Batch = typing.NewType('Batch', torch.Tensor)

ClassIndices = torch.LongTensor
OneHot = torch.FloatTensor
MultiLabelOneHot = OneHot

ImageTensor = torch.ShortTensor
NormalizedImageTensor = torch.FloatTensor

RGB = ImageTensor
Grayscale = ImageTensor
