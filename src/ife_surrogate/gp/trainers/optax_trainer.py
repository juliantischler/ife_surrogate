# ------------------------------------------------------------------------------
# Project: IFE_Surrogate
# Authors: Tobias Leitgeb, Julian Tischler
# CD Lab 2025
# ------------------------------------------------------------------------------
from ife_surrogate.gp.trainers import Trainer
from ife_surrogate.gp.models import GPModel

from functools import partial
import typing
from typing import Callable, Dict, Tuple, Optional
import optax
from jax import value_and_grad, jit, random
from typing import Tuple, Dict, Callable
from jaxtyping import Key, Array, Float, Int, Bool
import jax.numpy as jnp
import jax.random as jr


TypeGPModel = typing.TypeVar("GPModel", bound=GPModel)


class OptaxTrainer(Trainer):
    r"""
    Trainer for Gaussian Process models using gradient-based optimization with Optax[1].

    This class leverages JAX and Optax to optimize model parameters using
    gradient descent, supporting multiple restarts and early stopping.

    Attributes:
        key (Key): JAX random key for parameter sampling and reproducibility.
        optimizer: An Optax optimizer instance (e.g., optax.adam).[2]
        number_iterations (int): Maximum number of iterations per training run.
        number_restarts (int): Number of independent training restarts.
        tolerance (float): Minimum improvement threshold to reset early stopping.
        patience (int): Number of iterations to wait without improvement before stopping.
        verbose (bool): Whether to print iteration progress.
        save_history (bool): Whether to return the full optimization history.
        sample_parameters (bool): Whether to sample kernel hyperparameters at each restart.

    Methods:
        train(model: GPModel) -> Tuple[Dict, Optional[Dict]]:
            Trains the given GP model and returns the best parameters and optional history.
        info():
            Prints the current trainer settings for easy reference.
    [1] "https://optax.readthedocs.io/en/latest/"
    [2] "https://optax.readthedocs.io/en/latest/api/optimizers.html"
    """
    
    def __init__(self, 
                 key: Key = jr.key(seed=0),
                 optimizer = optax.adam(1e-3),
                 number_iterations: int = 100,
                 number_restarts: int = 1,
                 tolerance: Float = 1e-2, 
                 patience: Float = 20, **kwargs):
        super().__init__(**kwargs)
        self.key = key
        self.optimizer = optimizer
        self.tolerance = tolerance
        self.patience = patience
        self.number_iterations = number_iterations
        self.number_restarts = number_restarts

    def info(self):
        """Prints current settings of the OptaxTrainer."""
        print("OptaxTrainer settings:")
        print(f"\tkey: {self.key}")
        print(f"\toptimizer: {self.optimizer}")
        print(f"\tnumber_iterations: {self.number_iterations}")
        print(f"\tnumber_restarts: {self.number_restarts}")
        print(f"\ttolerance: {self.tolerance}")
        print(f"\tpatience: {self.patience}")
        print(f"\tverbose: {self.verbose}")
        print(f"\tsave_history: {self.save_history}")
        print(f"\tsample_parameters: {self.sample_parameters}")

    def train(self, model: TypeGPModel) -> Tuple[Dict, Optional[Dict]]:
        self.model = model
        value_and_grad_fn = value_and_grad(self.loss_fn)

        @jit
        def step(params: Dict, opt_state: Dict):
            loss, grads = value_and_grad_fn(params)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        def loop(params, opt_state):
            best_loss = jnp.inf
            no_improve = 0
            for it in range(self.number_iterations):
                params, opt_state, loss = step(params, opt_state)

                if loss + self.tolerance < best_loss:
                    best_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1

                if self.verbose and it % 50 == 0:
                    print(f"Iter {it}: loss={loss:.5f}, best={best_loss:.5f}")

                if no_improve >= self.patience:
                    if self.verbose:
                        print(f"Early stop at iter {it}, best loss {best_loss:.5f}")
                    break
            return params, opt_state, best_loss

        results = {}
        for r in range(self.number_restarts):
            _, self.key = random.split(self.key)
            if self.sample_parameters:
                params = self.model.kernel.sample_hyperparameters(self.key)
            else:
                params = self.model.kernel.get_params()

            opt_state = self.optimizer.init(params)
            params, opt_state, loss = loop(params, opt_state)

            results[f"run_{r}"] = {"params": params, "loss": loss, "key": self.key}
            if self.verbose:
                print(f"Run {r} loss: {loss}")

        results = {k: v for k, v in results.items() if not jnp.isnan(v["loss"])}
        best_key = min(results, key=lambda x: results[x]["loss"])

        if self.verbose:
            print("Best run:", results[best_key]["loss"])

        if self.save_history:
            return results[best_key], results
        return results[best_key], results


