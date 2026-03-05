#  IFE Surrogate GP

A flexible and extensible library for Gaussian Processes, built with performance and modularity in mind.  



---

##  Features

-  **High-performance kernels** (JAX-compatible)  
-  **Composable API** for building custom models and kernels
-  **Multiple optimizers** (Optax, PSO, etc.)  
-  **Built in Baysian inference of the models with NumPyro**
-  **Automatic hyperparameter handling**  
-  **Built-in training workflows**  

---

##  Installation

> Since we make heavy use of the JAX library this is not the most reliable way to a successful installation. For a full installation guide visit our [docs]( https://ife-surrogate.readthedocs.io/en/latest/index.html).



```bash
pip install ife_surrogate
```


---



## Quickstart

```python
from ife_surrogate.gp.kernels import Kriging
from ife_surrogate.gp.models import WidebandGP
from ife_surrogate.gp.trainers import SwarmTrainer


## Your dataset
dataset = np.load("some_data.npy", allow_pickle=True).item()
X, Y, f = dataset["X"], dataset["Y"], dataset["f"]

key = jr.key(seed=42)
(X_train, Y_train), (X_test, Y_test), _= train_test_split(
    X=X, Y=Y, f=f, 
    split=(0.9, 0.1, 0), 
    key= key
)

d = X_train.shape[1]
priors = {"lengthscale": Uniform(1e-0, 1e1), "power": Uniform(1, 2)}
kernel = kernels.Kriging(lengthscale=jnp.ones(d), power=jnp.ones(d), priors=priors)

model = models.WidebandGP(X_train, Y_train, kernel, f)

trainer = trainers.SwarmTrainer(number_iterations=200, number_particles=10)
trainer.train(model)

pred, var = model.predict(X_test)
```
---

##  Documentation

Visit our full [documentation](https://ife-surrogate.readthedocs.io/en/latest/index.html).

---

##  Key Components

- **Kernels**  
  - Kriging
  - RBF
  - Matern
  - Scale
  - Noise
  - SumKernel
  - ProductKernel

- **Models**  
  - Wideband Gaussian Process Model
  - Scalar Gaussian Process Model



- **Trainers**  
  - OptaxTrainer (Trainer class leveraging JAX for speedy training)  
  - SwarmTrainer (TRainer class using Particle Swarm Optimization)

---

##  Roadmap

  - Wideband Student-t Process Model
  - Scalar Student-t Process Model

---

##  License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.  




## References

The mathematical background and implementation of these models are based on the following publications:

### Gaussian Process & Student-t Theory

* **Gaussian Process Basics**: Rasmussen, C. E., & Williams, C. K. I. (2006). [Gaussian Processes for Machine Learning](http://www.GaussianProcess.org/gpml). MIT Press.
* **Student-t Process (TP) Foundations**: Shah, A., Wilson, A. G., & Ghahramani, Z. (2014). [Student-t Processes as Alternatives to Gaussian Processes](https://www.cs.cmu.edu/~andrewgw/tprocess.pdf). *Proceedings of the 17th International Conference on Artificial Intelligence and Statistics (AISTATS)*. 

### Wideband & Multi-output Modeling

* **Wideband Architecture**: Rezende, R. S., Hansen, J., Piwonski, A., & Schuhmann, R. (2024). Wideband Kriging for Multiobjective Optimization of a High-Voltage EMI Filter. *IEEE Transactions on Electromagnetic Compatibility*, 66(4), 1116–1124. 
* **Multi-output Separable GPs**: Bilionis, I., Zabaras, N., Konomi, B. A., & Lin, G. (2013). Multi-output separable Gaussian process: Towards an efficient, fully Bayesian paradigm for uncertainty quantification. *Journal of Computational Physics*, 241, 212–239. 
* **Vector-Valued Kernels**: Alvarez, M. A., Rosasco, L., & Lawrence, N. D. (2012). [Kernels for Vector-Valued Functions: A Review](https://arxiv.org/abs/1106.6251). [cite_start]*Foundations and Trends in Machine Learning*.

### Inference & Software Stack

* **JAX**: Bradbury, J., et al. (2018). [JAX: Composable transformations of Python+NumPy programs](http://github.com/jax-ml/jax).
* **NumPyro**: Phan, D., Pradhan, N., & Jankowiak, M. (2019). [Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro](https://arxiv.org/abs/1912.11554). *arXiv preprint*.
* **Flax**: Heek, J., et al. (2020). [Flax: A neural network library and ecosystem for JAX](https://github.com/google/flax).
* **Optax**: Babuschkin, I., et al. (2020). [Optax: Composable gradient descent optimization for JAX](https://github.com/google-deepmind/optax).
* **SciPy**: Virtanen, P., et al. (2020). [SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python](https://www.nature.com/articles/s41592-019-0686-2). *Nature Methods*.
* **NUTS Sampler**: Hoffman, M. D., & Gelman, A. (2014). [The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo](https://jmlr.org/papers/v15/hoffman14a.html). *Journal of Machine Learning Research*. 

## Acknowledgements
