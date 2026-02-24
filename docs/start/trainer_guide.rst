Trainer Guide
=================

The ``Trainer`` represents the final stage of the intended ``ife-surrogate`` workflow. While the **Kernel** (such as ``RBF``) defines the spatial relationship of the data and the **Model** (such as ``WidebandGP``) manages the data and likelihood structures, the **Trainer** is responsible for the numerical execution of the learning process.

Decoupling the Trainer from the Model allows us to apply different optimization philosophies, from gradient descent to global heuristic searches, without changing the underlying GP definition.

The Three-Step Workflow
^^^^^^^^^^^^^^^^^^^^^^^^

To train a surrogate, we suggest the following pipeline:

1.  **Kernel Definition**: Select and compose kernels (e.g., ``Matern12``, ``RBF``, or additive compositions via kernel operators like ``ProductKernel``) to define the prior covariance.
2.  **Model Assembly**: Initialize a specific Model class with a Kernel and some training data. The model defines the objective function (the Marginal Log-Likelihood) that the trainer will minimize.
3.  **Execution via Trainer**: Pass the Model into a specialized Trainer class. 


Depending on the complexity of the loss landscape, one can choose from various trainer implementations:

* **OptaxTrainer**: Leverages the ``optax`` library for gradient-based optimization. Ideal for high-dimensional parameter spaces where Adam, SGD, or multi-stage learning rate schedules are required.
* **SwarmTrainer**: A global optimization approach using Particle Swarm Optimization (PSO). This is particularly useful for non-convex likelihood surfaces where gradient descent might get stuck in local minima.

Each trainer exposes specific hyperparameters (such as swarm size, inertia weights, or optimizer strategies) allowing for fine-grained control over the convergence behavior.
  

.. note::
   Ensure your data is appropriately scaled before passing it to the ``Trainer`` to improve numerical stability during the JAX optimization process.

----



OptaxTrainer
^^^^^^^^^^^^^^^^^^^^

The OptaxTrainer class leans heavily on the `optax <https://optax.readthedocs.io/en/latest/index.html>`_ library a comprehensive optimization library using jax. 
Their page on `optimizers <https://optax.readthedocs.io/en/latest/api/optimizers.html>`_ is particularly helpful here.


Usage:
------



----


SwarmTrainer
^^^^^^^^^^^^^^^^^^^^

Technical Overview:
-------------------

The foundational mathematical model for swarm training is
**Particle Swarm Optimization (PSO)**.

Let :math:`S` be a swarm of :math:`N` particles (candidate solutions or models).
Each particle :math:`i` has a position vector
:math:`x_i \in \mathbb{R}^d` representing the model weights or hyperparameters,
and a velocity vector :math:`v_i`.

The objective is to minimize a loss function :math:`f(x)`.


At iteration :math:`t`, each particle tracks:

- **Current Position:** :math:`x_i^{(t)}`
- **Current Velocity:** :math:`v_i^{(t)}`
- **Personal Best:** The best position found by this specific particle so far.

  .. math::

     p_{i,\text{best}} = \arg\min_{\tau = 1 \ldots t} f\!\left(x_i^{(\tau)}\right)

- **Global Best:** The best position found by the entire swarm.

  .. math::

     g_{\text{best}} = \arg\min_{j = 1 \ldots N} f\!\left(p_{j,\text{best}}\right)


The particles update their trajectory based on inertia, cognitive influence
(memory), and social influence (collective knowledge). The velocity update
equation is:

.. math::

   v_i^{(t+1)} =
   \underbrace{w \cdot v_i^{(t)}}_{\text{Inertia}}
   + \underbrace{c_1 r_1 \cdot \left(p_{i,\text{best}} - x_i^{(t)}\right)}_{\text{Cognitive}}
   + \underbrace{c_2 r_2 \cdot \left(g_{\text{best}} - x_i^{(t)}\right)}_{\text{Social}}

where:

- :math:`w` is the inertia weight (controls exploration vs. exploitation)
- :math:`c_1, c_2` are acceleration coefficients
- :math:`r_1, r_2 \sim U(0,1)` are random stochastic factors

The position is then updated as:

.. math::

   x_i^{(t+1)} = x_i^{(t)} + v_i^{(t+1)}


Usage:
---------------------




----
