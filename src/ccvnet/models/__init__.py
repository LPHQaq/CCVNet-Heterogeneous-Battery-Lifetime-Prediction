"""Model definitions for ccvnet and baselines."""

from ccvnet.models.cnn import CNNBaseline
from ccvnet.models.ccvnet import ccvnet
from ccvnet.models.mlp import MLPBaseline
from ccvnet.models.spectrumnn import SpectrumNN

__all__ = ["ccvnet", "CNNBaseline", "MLPBaseline", "SpectrumNN"]

