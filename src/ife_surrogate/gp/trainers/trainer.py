# ------------------------------------------------------------------------------
# Project: IFE_Surrogate
# Authors: Tobias Leitgeb, Julian Tischler
# CD Lab 2025
# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod
import typing
from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple, Optional
from functools import partial
from ..models import GPModel


TypeGPModel = typing.TypeVar("GPModel", bound=GPModel)


class Trainer(ABC):
    def __init__(
        self,
        sample_parameters: bool = True,
        verbose: bool = False,
        save_history: bool = False
    ):
        self.sample_parameters = sample_parameters
        self.verbose = verbose
        self.save_history = save_history
        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model
        likelihood = self.model.likelihood
        if "sigma_sq" in self.model.get_attributes().keys():
            self.nlml = partial(likelihood, self.model.X, self.model.Y, self.model.sigma_sq, self.model.jitter)
        else:
            self.nlml = partial(likelihood, self.model.X, self.model.Y, self.model.jitter)


    def loss_fn(self, updated_params: Dict) -> float:
        return self.nlml(self.model.kernel)

    @abstractmethod
    def train(self, *args, **kwargs) -> Tuple[Dict, Optional[Dict]]:
        """Run optimization and return (best_run, history)."""
        pass