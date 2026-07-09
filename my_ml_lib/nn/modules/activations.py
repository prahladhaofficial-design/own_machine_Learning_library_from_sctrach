# In my_ml_lib/nn/modules/activations.py
from .base import Module
# Assuming Value class has .relu() and other necessary methods implemented

class ReLU(Module):
    """Applies the Rectified Linear Unit function element-wise."""

    def __call__(self, x):
        """
        Forward pass: Applies ReLU activation.

        Args:
            x (Value): Input Value object.

        Returns:
            Value: Output Value object (result of x.relu()).
        """
        # Call the relu method on the input Value object
        return x.relu()

    def __repr__(self):
        return "ReLU()"

class Sigmoid(Module):
    """Applies the Sigmoid function element-wise."""

    def __call__(self, x):
        """
        Forward pass: Applies Sigmoid activation.

        Args:
            x (Value): Input Value object.

        Returns:
            Value: Output Value object after applying sigmoid.
        """
        # Implement sigmoid using the formula: 1 / (1 + exp(-x))
        # This uses the Value operations we already defined (__add__, __neg__, exp, __pow__ for division)
        return (1 + (-x).exp())**-1

    def __repr__(self):
        return "Sigmoid()"
