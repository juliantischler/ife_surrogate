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
--------------

Get up and running in seconds with pip.

.. code-block:: bash

	pip install ife-surrogate

Note that in order to increase performance we rely on the JAX module. Instructions on how to install JAX can be found `here <https://github.com/jax-ml/jax?tab=readme-ov-file#installation>`_.


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
	start/creating_custom_kernels
	start/creating_custom_models
	start/guide_to_trainers


.. toctree::
	:maxdepth: 3
	:caption: API Reference
	:hidden:

	api/gp
	api/gp/kernels
	api/gp/models
	api/gp/trainers
	api/utils
	