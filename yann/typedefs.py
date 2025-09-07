import typing

import torch

Logits = torch.FloatTensor
Embedding = torch.FloatTensor
OneHot = torch.Tensor

Probabilities = torch.FloatTensor
LogProbabilities = torch.FloatTensor

Batch = typing.NewType('Batch', torch.Tensor)

ClassIndices = torch.LongTensor
MultiLabelOneHot = OneHot

ImageTensor = torch.ShortTensor
NormalizedImageTensor = torch.FloatTensor

RGB = ImageTensor
Grayscale = ImageTensor
