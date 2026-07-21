"""Compatibility exports for PyTorch stochastic weight averaging utilities."""

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

__all__ = ['AveragedModel', 'SWALR', 'update_bn']
