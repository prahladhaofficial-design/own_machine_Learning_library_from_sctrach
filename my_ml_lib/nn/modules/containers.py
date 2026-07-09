# my_ml_lib/nn/modules/containers.py

from .base import Module
from collections import OrderedDict
from ..autograd import Value

class Sequential(Module):
    """
    A sequential container for stacking Modules.
    Modules will be executed in the order they are added.
    """

    def __init__(self, *args):
        """
        Initializes the Sequential container.

        Args:
            *args (Module): Sequence of modules to add.
                            Can also be an OrderedDict.
        """
        # Initialize base class and register modules
        super().__init__()  # CRITICAL: Initializes _parameters and _modules

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            # If an OrderedDict is passed, add modules using keys as names
            for key, module in args[0].items():
                if not isinstance(module, Module):
                    raise TypeError(f"Value for key '{key}' is not an nn.Module subclass: {type(module)}")
                self.add_module(key, module)
        else:
            # If positional arguments are passed, add modules using indices as names
            for i, module in enumerate(args):
                if not isinstance(module, Module):
                    raise TypeError(f"Argument {i} is not an nn.Module subclass: {type(module)}")
                self.add_module(str(i), module)

    def __call__(self, x: Value) -> Value:
        """
        Defines the forward pass through the sequential layers.

        Args:
            x (Value): Input Value object.

        Returns:
            Value: Output of the final layer in the sequence.
        """
        # Iterate through the modules stored in this container
        # and pass the output of one module as the input to the next.
        for module in self._modules.values():
            x = module(x)
        
        return x

    def __repr__(self):
        """Provides a developer-friendly string representation of the container."""
        # This iterates through the registered modules and their names
        layer_strs = [f"  ({name}): {module}" for name, module in self._modules.items()]
        return f"Sequential(\n" + "\n".join(layer_strs) + "\n)"
