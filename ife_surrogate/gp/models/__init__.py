# ------------------------------------------------------------------------------
# Project: IFE_Surrogate
# Authors: Tobias Leitgeb, Julian Tischler
# CD Lab 2025
# ------------------------------------------------------------------------------
"""
Contains all Gaussian Process models of the Library.
"""
from .gp_model import GPModel
from .scalar_gp import ScalarGP, ScalarGPBaysian
from .wideband_gp import WidebandGP, WidebandGPBaysian
from .multi_output_gp import SeparableMultiOutputGP


__all__ = [
    "GPModel",
    "WidebandGP",  
    "WidebandGPBaysian",
    "ScalarGP",
    "ScalarGPBaysian",
    "SeparableMultiOutputGP",
    ]