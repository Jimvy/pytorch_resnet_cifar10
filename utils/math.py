r"""
Polymporphic maths :o
Because torch's math functions kind of suck...
"""

import torch
import numpy


def sqrt(x):
    if isinstance(x, torch.Tensor):
        return torch.sqrt(x)
    else:
        return numpy.sqrt(x)

