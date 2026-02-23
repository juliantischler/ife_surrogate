# ------------------------------------------------------------------------------
# Project: IFE_Surrogate
# Authors: Tobias Leitgeb, Julian Tischler
# CD Lab 2025
# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple, List, Set
from jaxtyping import Array, Key
from flax import struct



@struct.dataclass
class Kernel(ABC):
    """
    Abstract base class for kernels in Gaussian Process (GP) regression.

    This class provides the core interface and utility functions for all kernel
    implementations, including methods to evaluate the kernel, access and
    update parameters, and sample new hyperparameters from prior distributions.

    Attributes:
        priors (Dict[str, Callable]): A dictionary mapping parameter names to 
            callable prior distributions used for sampling hyperparameters.
        param_bounds (Dict, optional): Dictionary specifying bounds for each 
            parameter. Defaults to an empty dictionary.
        exempt (Set[str]): Set of attribute names to exclude when retrieving 
            parameters (e.g., priors, param_bounds, exempt itself).

    Methods:
        evaluate(x1, x2):
            Abstract method. Computes the kernel function between two inputs.
        __call__(x1, x2):
            Calls `evaluate`, allowing the kernel object to be used as a function.
        get_priors():
            Returns the dictionary of priors for kernel hyperparameters.
        get_params():
            Returns a dictionary of the current kernel parameters, excluding
            those in `exempt`.
        sample_hyperparameters(key):
            Samples new parameter values from the prior distributions using
            a JAX random key.
        update_params(params):
            Updates kernel parameters in-place using the provided dictionary.
    """

    priors: Dict[str, Callable] = struct.field(pytree_node=False, kw_only=True)
    param_bounds: Dict = struct.field(pytree_node=False, default_factory=dict, kw_only=True)
    exempt: Set[str] = struct.field(
        pytree_node=False,
        default_factory=lambda: {'priors', 'param_bounds', 'exempt'},
        kw_only=True
    )
    
    @abstractmethod
    def evaluate(self, x1: Array, x2: Array) -> Array:
        """
        Evaluate the kernel function.

        Args:
            x1: First input data array.
            x2: Second input data array.

        Returns:
            Kernel evaluation between x1 and x2.
        """
        pass

    def __call__(self, x1: Array, x2: Array) -> Array:
        """
        Args:
            x1: First input data array.
            x2: Second input data array.

        Returns:
            Kernel evaluation between x1 and x2.
        """
        return self.evaluate(x1, x2)
    
    def get_priors(self):
        """
        Get the kernel parameters.

        Returns:
            A dictionary containing the kernel parameters.
        """
        return self.priors
    
    def get_params(self) -> Dict[str, Array]:
        """
        Get the kernel parameters.

        Returns:
            A dictionary containing the kernel parameters.
        """
        return {x: y for x, y in self.__dict__.items() if x not in self.exempt}
        

    def sample_hyperparameters(self, key: Key) -> Dict[str, Array]:
        """
        Sample new parameters from the prior distribution.

        Args:
            key: JAX random key.

        Returns:
            A dictionary containing sampled parameters.
        """
        assert isinstance(key, Array), "'key' must be a jax.random key"
        if self.priors is None:
            print("No priors specified. Returning current parameters.")
            return self.get_params()

        param_keys = self.get_params().keys()
        
        param_samples = {}
        for x in param_keys:
            if x in self.priors:
                param_samples[x] = self.priors[x].sample(key, self.__dict__[x].shape)
            else:
                param_samples[x] = self.__dict__[x]
        
        return param_samples


    def update_params(self, params: Dict[str, Array]) -> None:
        """
        Update the kernel parameters.

        Args:
            params: A dictionary containing the new parameters.
        """

        for key in params:
            self.__dict__[key] = params[key]

    
