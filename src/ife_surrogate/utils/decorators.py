# ------------------------------------------------------------------------------
# Project: IFE_Surrogate
# Authors: Tobias Leitgeb, Julian Tischler
# CD Lab 2025
# ------------------------------------------------------------------------------
import warnings
import functools

class ExperimentalWarning(UserWarning):
    """Warning for experimental API usage."""
    pass


def experimental(obj):
    """
    Decorator to mark functions/classes as experimental.

    Emits a warning on first use.
    """
    message = f"{obj.__name__} is experimental and may change or be removed in future versions."

    if isinstance(obj, type):
        ## Decorating a class
        orig_init = obj.__init__

        @functools.wraps(orig_init)
        def new_init(self, *args, **kwargs):
            warnings.warn(message, ExperimentalWarning, stacklevel=2)
            return orig_init(self, *args, **kwargs)

        obj.__init__ = new_init
        obj.__experimental__ = True
        return obj

    else:
        ## Decorating a function
        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(message, ExperimentalWarning, stacklevel=2)
            return obj(*args, **kwargs)

        wrapper.__experimental__ = True
        return wrapper



class DeprecatedWarning(UserWarning):
    """Warning for deprecated API usage."""
    pass


def deprecated(deprecated_in=None, removed_in=None, use_instead=None):
    """
    Decorator to mark a function/class as deprecated.

    Parameters
    ----------
    deprecated_in : str, optional
        Version in which it was deprecated.
    removed_in : str, optional
        Version in which it will be removed.
    use_instead : str, optional
        Name of the recommended replacement.
    """
    def decorator(obj):
        msg = f"{obj.__name__} is deprecated"
        if deprecated_in:
            msg += f" since version {deprecated_in}"
        if removed_in:
            msg += f" and will be removed in version {removed_in}"
        if use_instead:
            msg += f". Use {use_instead} instead."
        msg += "."

        if isinstance(obj, type):
            ## classes
            init = obj.__init__

            @functools.wraps(init)
            def new_init(self, *args, **kwargs):
                warnings.warn(msg, DeprecatedWarning, stacklevel=2)
                return init(self, *args, **kwargs)

            obj.__init__ = new_init
            obj.__deprecated__ = True
            return obj

        else:
            ## functions
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                warnings.warn(msg, DeprecatedWarning, stacklevel=2)
                return obj(*args, **kwargs)

            wrapper.__deprecated__ = True
            return wrapper

    return decorator
