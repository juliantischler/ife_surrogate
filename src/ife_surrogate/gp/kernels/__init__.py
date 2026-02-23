# ------------------------------------------------------------------------------
# Project: IFE_Surrogate
# Authors: Tobias Leitgeb, Julian Tischler
# CD Lab 2025
# ------------------------------------------------------------------------------
"""
Contains all internal kernels for the GPR.
New Kernels can be created by inheriting from the base kernel 'Kernel'

Example:
    >>> from ife_surrogate.gp.kernels import Kernel
    >>> from flax import struct
    >>> 
    >>> @struct.dataclass
    >>> class NewKernel(Kernel):
    >>>     def evaluate(self, x1: Array, x2: Array) -> Array:
    >>>         # some kernel definition
    >>>         return []
"""
from .kernel import Kernel
from .rbf import RBF
from .kriging_kernel import Kriging
from .rational_quadratic import RQ
from .matern import Matern12, Matern32, Matern52
from .compound_kernels import SumKernel, ProductKernel, SeparableKernel
from .scale import Scale
from .noise import DiagNoise
from .fixed_kernel import FixedFreqKernel


__all__ = [
    'Kernel',
    'RBF', 
    'Kriging', 
    'RQ',
    'Matern12',
    'Matern32',
    'Matern52',
    'SumKernel',
    'ProductKernel',
    'Scale',
    'DiagNoise',
    'SeparableKernel',
    'FixedFreqKernel'
]