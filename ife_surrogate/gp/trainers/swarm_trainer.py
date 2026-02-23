# ------------------------------------------------------------------------------
# Project: IFE_Surrogate
# Authors: Tobias Leitgeb, Julian Tischler
# CD Lab 2025
# ------------------------------------------------------------------------------
from ife_surrogate.gp.trainers import Trainer
from ife_surrogate.gp.models import GPModel

from jax import vmap, jit
import typing
import pyswarms as ps
from jax.flatten_util import ravel_pytree
import numpy as np
from jaxtyping import Array

import jax.numpy as jnp
from jax import jit, vmap
from jax.flatten_util import ravel_pytree
import numpy as np
import pyswarms as ps
from flax.core.frozen_dict import FrozenDict
import jax.tree_util as tree_util

TypeGPModel = typing.TypeVar("GPModel", bound=GPModel)


class SwarmTrainer(Trainer):
    r"""
    Trainer class that optimizes model parameters using Particle Swarm Optimization (PSO) via pyswarms[1]

    This class extends a generic `Trainer` and implements a global optimization
    method for training models, particularly useful when gradients are unavailable
    or the loss landscape is highly non-convex.

    Args:
        swarm_settings (dict, optional): Dictionary of PSO hyperparameters:
            - `c1` (float): Cognitive parameter (default: 0.5)
            - `c2` (float): Social parameter (default: 0.3)
            - `w` (float): Inertia weight (default: 0.9)
        number_iterations (int, optional): Number of iterations for the PSO algorithm per restart (default: 100).
        number_particles (int, optional): Number of particles in the swarm (default: 20).
        number_restarts (int, optional): Number of independent PSO runs to avoid local minima (default: 1).
        bounds (dict[str: (float, float)], optional): Lower and upper bounds for each parameter. 
            Not defined parameter bounds are automatically inferred from kernel priors.
        **kwargs: Additional keyword arguments passed to the base `Trainer` class.

    Attributes:
        swarm_settings (dict): Hyperparameters controlling particle behavior in PSO.
        number_iterations (int): Iterations per PSO run.
        number_particles (int): Number of particles in the swarm.
        number_restarts (int): Number of independent PSO runs.
        bounds (dict[str: (float, float)]): Parameter bounds.
        model: The model to be optimized.
        verbose (bool): Flag to control printing of optimization progress (inherited from `Trainer`).

    Methods:
        info(): Prints current PSO settings and bounds.
        train(model): Optimizes the provided model's kernel parameters using PSO and returns
                      the best parameters and optionally the optimization history.

    Example:
        >>> trainer = SwarmTrainer(number_particles=30, number_iterations=200)
        >>> best_run, history = trainer.train(model)
    
    [1] 'https://github.com/ljvmiranda921/pyswarms'
    """

    def __init__(self, 
                 swarm_settings={"c1":1.5, "c2":1.5, "w":0.5},
                 number_iterations: int = 100,
                 number_particles: int = 20,
                 number_restarts: int = 1,
                 bounds: dict = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.swarm_settings = swarm_settings
        self.number_iterations = number_iterations
        self.number_particles = number_particles
        self.number_restarts = number_restarts
        self.bounds = bounds if bounds else {}

    def info(self):
        print("SwarmTrainer settings: ")
        print("\tswarm_settings: ", self.swarm_settings)
        print("\tnumber_iterations: ", self.number_iterations)
        print("\tnumber_particles: ", self.number_particles)
        print("\tnumber_restarts: ", self.number_restarts)
        print("\tbounds: ", self.bounds)

    def train(self, model):
        self.model = model
        kernel = model.kernel
        nlml = self.nlml

        params_template = kernel.get_params()
        flat_template, unravel_fn = ravel_pytree(params_template)
        dim = len(flat_template)

        ## Generate bounds automatically, if not defined
        priors = kernel.get_priors()
        keys = params_template.keys()
        for name in priors.keys():
            if name in keys and name not in self.bounds.keys():
                args = priors[name].get_args()
                low = args.get("low")
                high = args.get("high")
                self.bounds[name] = [max(0.0, low), max(0.0, high)]

        _bounds = ([], [])
        for name, __bounds in self.bounds.items():
            if name in params_template.keys():
                _lb = [__bounds[0]] * len(params_template[name])
                _ub = [__bounds[1]] * len(params_template[name])
                _bounds[0].extend(_lb)
                _bounds[1].extend(_ub)

        ## function for raveling and unraveling the hyperparameter dictionary of the kernel
        params_template = kernel.get_params()
        raveled_template, unravel_fn = ravel_pytree(params_template)        
        
        # def single_particle_loss_(flat_params: Array) -> float:
        #     #params = array_to_dict(params)
        #     params = unravel_fn(flat_params)
        #     kernel.update_params(params)
        #     return nlml(kernel)

        def single_particle_loss_(flat_params: Array) -> float:
        
            params = unravel_fn(flat_params)
            kernel.update_params(params)
            # CRITICAL FIX: Use .replace() instead of .update_params()
            # Flax dataclasses are immutable. .replace() creates a new instance safely.
            # This bypasses the in-place mutation bug in Kernel.update_params.
            updated_kernel = kernel.replace(**params)
            
            # Pass the NEW kernel instance to the loss function
            return nlml(updated_kernel)
        
        single_particle_loss = jit(vmap(single_particle_loss_))
        # single_particle_loss = single_particle_loss_
        
        history = {}
        for r in range(self.number_restarts):
            optimizer = ps.single.GlobalBestPSO(
                n_particles=self.number_particles,
                dimensions=dim,
                options=self.swarm_settings,
                bounds=_bounds
            )

            best_cost, best_pos = optimizer.optimize(
                single_particle_loss,
                iters=self.number_iterations,
                verbose=True
            )
            
            best_params = unravel_fn(best_pos)
            kernel.update_params(best_params)

            history[f"run_{r}"] = {
                "params": best_params,
                "loss": best_cost
            }
            if self.verbose:
                print(f"Run {r}: Best loss = {best_cost:.5f}")

        ## Remove nans, there should not be any though
        history = {k: v for k, v in history.items() if not np.isnan(v["loss"])}
        best_key = min(history, key=lambda k: history[k]["loss"])

        if self.verbose:
            print("Overall best run:", best_key, "loss:", history[best_key]["loss"])

        best_run = history[best_key]
        return (best_run, history) if self.save_history else (best_run, {})
    
    # def train(self, model):
    #     print("New train fn")
    #     self.model = model
    #     kernel = model.kernel
    #     nlml = self.nlml

    #     # 1. Get the template and flattening function
    #     params_template = kernel.get_params()
    #     flat_template, unravel_fn = ravel_pytree(params_template)
    #     dim = len(flat_template)

    #     # 2. Infer bounds if not present
    #     priors = kernel.get_priors()
    #     param_keys = params_template.keys() if hasattr(params_template, 'keys') else []
        
    #     for name in priors.keys():
    #         if name in param_keys and name not in self.bounds:
    #             args = priors[name].get_args()
    #             low = args.get("low")
    #             high = args.get("high")
    #             l_val = max(0.0, low) if low is not None else 0.0
    #             h_val = max(0.0, high) if high is not None else 1e5 
    #             self.bounds[name] = [l_val, h_val]

    #     # 3. Construct _bounds vector aligned with flat_template
    #     # FIX: We construct bounds by mapping over the parameter structure, not the bounds dict
        
    #     def get_bounds_for_leaf(path, leaf_val):
    #         # path is a tuple of keys/indices. path[0] is usually the param name.
    #         if not path:
    #             return (-jnp.inf, jnp.inf)
            
    #         # Extract key name safely
    #         key = str(path[0].key) if hasattr(path[0], 'key') else str(path[0])

    #         if key in self.bounds:
    #             b_low, b_high = self.bounds[key]
    #             # Ensure bounds match the shape of the parameter array
    #             return (jnp.full(leaf_val.shape, b_low), jnp.full(leaf_val.shape, b_high))
    #         else:
    #             return (jnp.full(leaf_val.shape, -np.inf), jnp.full(leaf_val.shape, np.inf))

    #     # Generate a PyTree of bounds matching the structure of params_template
    #     bounds_tree = tree_util.tree_map_with_path(get_bounds_for_leaf, params_template)
        
    #     # Flatten the bounds tree using the same unravel logic as the parameters
    #     flat_lb, _ = ravel_pytree(tree_util.tree_map(lambda x: x[0], bounds_tree))
    #     flat_ub, _ = ravel_pytree(tree_util.tree_map(lambda x: x[1], bounds_tree))
        
    #     _bounds = (np.array(flat_lb), np.array(flat_ub))

    #     # 4. Define Loss Function
    #     def single_particle_loss_(flat_params):
    #         params = unravel_fn(flat_params)
            
    #         # CRITICAL FIX: Use .replace() instead of .update_params()
    #         # Flax dataclasses are immutable. .replace() creates a new instance safely.
    #         # This bypasses the in-place mutation bug in Kernel.update_params.
    #         # We also ensure params is strictly a dict for unpacking.
    #         if not isinstance(params, dict):
    #              # This handles cases where unravel_fn might return a different PyTree structure
    #              # though ravel_pytree on a dict template should return a dict.
    #              pass

    #         updated_kernel = kernel.replace(**params)
            
    #         # Pass the NEW kernel instance to the loss function
    #         return nlml(updated_kernel)
            
    #     # JIT and VMAP for efficiency
    #     single_particle_loss = jit(vmap(single_particle_loss_))
        
    #     history = {}
    #     for r in range(self.number_restarts):
    #         optimizer = ps.single.GlobalBestPSO(
    #             n_particles=self.number_particles,
    #             dimensions=dim,
    #             options=self.swarm_settings,
    #             bounds=_bounds
    #         )

    #         # optimize() takes a function that accepts (N_particles, dimensions)
    #         best_cost, best_pos = optimizer.optimize(
    #             single_particle_loss,
    #             iters=self.number_iterations,
    #             verbose=self.verbose
    #         )

    #         best_params = unravel_fn(best_pos)
            
    #         history[f"run_{r}"] = {
    #             "params": best_params,
    #             "loss": best_cost
    #         }
    #         if self.verbose:
    #             print(f"Run {r}: Best loss = {best_cost:.5f}")

    #     # Filter out failed runs (NaNs)
    #     history = {k: v for k, v in history.items() if not np.isnan(v["loss"])}
        
    #     if not history:
    #          print("Optimization failed: All runs resulted in NaN loss.")
    #          return params_template, {}

    #     best_key = min(history, key=lambda k: history[k]["loss"])

    #     if self.verbose:
    #         print("Overall best run:", best_key, "loss:", history[best_key]["loss"])

    #     best_run = history[best_key]
        
    #     # We return the best parameters. The caller is responsible for applying them 
    #     # to the model if desired, as we cannot mutate the frozen Kernel here.
    #     return best_run, history if self.save_history else best_run