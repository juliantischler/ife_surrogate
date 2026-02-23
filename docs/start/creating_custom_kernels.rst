Creating Custom kernels
=======================

Implementing your own kernels is very simple. IFE-Surrogate provides the base class 'Kernel' which we inherit. We write a kernel function in the 'evaluate' method of our custom kernel and define the parameters in the class. The kernel is now usable.


.. code-block:: python

    # my_kernel.py
    from flax import struct

    @struct.dataclass 
    class MyKernel(Kernel):
        r"""
        A custom kernel.
        """
        my_parameter: Array 


        def evaluate(self, x1: Array, x2: Array) -> Array:
            r"""
            A custom kernel function.
            """
            return (x1 + x2) * my_parameter