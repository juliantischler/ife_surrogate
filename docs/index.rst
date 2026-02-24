IFE Surrogate
=============

.. image:: https://img.shields.io/pypi/v/ife-surrogate.svg
	:target: https://pypi.org/project/ife-surrogate/
	:alt: PyPI Version

.. image:: https://img.shields.io/pypi/l/ife-surrogate.svg
	:target: https://pypi.org/project/ife-surrogate/
	:alt: License

.. image:: https://img.shields.io/python/required/ife-surrogate.svg
	:alt: Python Version
  
  

**IFE Surrogate** is a modular Gaussian Process library designed for surrogate model training of electric circuits. It provides a flexible interface for defining kernels, training models, and generating surrogate predictions with ease.

----

Installation
============

``ife-surrogate`` can be installed on Linux, macOS, and Windows. Because this library relies on **JAX** for high-performance computations, we recommend setting up your environment first to ensure hardware acceleration (GPU/TPU) is correctly configured.

Basic Installation (CPU)
------------------------
For standard use on a laptop or CPU-bound server, you can install everything via ``pip``:

.. code-block:: bash

    # Upgrade pip and install JAX (CPU version)
    pip install --upgrade pip
    pip install --upgrade "jax[cpu]"

    # Install the library
    pip install ife-surrogate

Conda Installation (Recommended)
--------------------------------
If you prefer managing your environment with **Conda** or **Mamba**, use the ``conda-forge`` channel which provides a community-supported JAX build:

.. code-block:: bash

    # Create a new environment
    conda create -n ife_env python=3.10
    conda activate ife_env

    # Install JAX from conda-forge
    conda install -c conda-forge jax

    # Install the library via pip
    pip install ife-surrogate

GPU Installation (NVIDIA)
-------------------------
To leverage NVIDIA GPUs, JAX requires specific CUDA and cuDNN versions. It is highly recommended to install JAX with GPU support **before** installing ``ife-surrogate``.

**Via Pip:**

.. code-block:: bash

    # For CUDA 12 support
    pip install --upgrade "jax[cuda12]"
    pip install ife-surrogate

**Via Conda:**

.. code-block:: bash

    # This installs jaxlib with the necessary CUDA toolkit
    conda install -c conda-forge "jaxlib=*=*cuda*" jax
    pip install ife-surrogate

Verify Installation
-------------------
After installing, you can verify that the library and JAX are seeing your hardware correctly:

.. code-block:: python

    import jax
    import ife_surrogate

    print(f"JAX version: {jax.__version__}")
    print(f"Devices detected: {jax.devices()}")

----


Support
--------------

 

----


License
--------------

The software is provided under the MIT License.

----

Index
--------------

* :ref:`genindex`



.. toctree::
	:maxdepth: 2
	:caption: User Guide
	:hidden:

	start/getting_started
	start/trainer_guide
	start/creating_custom_kernels
	start/creating_custom_models


.. toctree::
	:maxdepth: 3
	:caption: API Reference
	:hidden:

	api/gp/kernels
	api/gp/models
	api/gp/trainers
	api/utils
	