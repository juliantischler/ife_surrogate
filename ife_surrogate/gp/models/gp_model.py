# ------------------------------------------------------------------------------
# Project: IFE_Surrogate
# Authors: Tobias Leitgeb, Julian Tischler
# CD Lab 2025
#
# Base class for Gaussian Process Regressor Models
#
# ------------------------------------------------------------------------------
from ..kernels import Kernel

from jaxtyping import Array, Key
import typing
from abc import ABC, abstractmethod
import joblib


TypeKernel = typing.TypeVar("Kernel", bound=Kernel)



class GPModel(ABC):
    r""" 
        Abstract base class of a GP, to be used as skeleton for actual GP implementations.
    """

    def __init__(self, kernel: TypeKernel, X: Array, Y: Array):
        self.kernel = kernel
        self.X = X
        self.Y = Y

    ## Must be implemented in actual model implementation, every model should have a predict method
    @abstractmethod
    def predict():
        return

    def get_attributes(self):
        return self.__dict__

    #TODO 
    def save(self, filename="IfeSurrModel.pkl"):
        with open(filename, "wb") as output:
            joblib.dump(self, filename)
    
    @staticmethod
    def load(filename):
        return joblib.load(filename)