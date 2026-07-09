# my_ml_lib/nn/autograd.py
import numpy as np
import math # For exp, log etc.

class Value:
    """
    Stores a scalar or numpy array and its gradient.
    Builds a computation graph for automatic differentiation (backpropagation).
    Inspired by micrograd: [https://github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
    """
    def __init__(self, data, _parents=(), _op='', label=''):
        """
        Initializes a Value object.

        Args:
            data: Input data (convertible to numpy float64 array).
            _parents (tuple): Parent Value objects that created this node.
            _op (str): Operation that created this node.
            label (str): Optional label for debugging/visualization.
        """
        # Data Type Conversion & Initialization ---
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data, dtype=np.float64)
            except TypeError:
                raise TypeError(f"Data must be convertible to a numpy array, got {type(data)}")
        if not np.issubdtype(data.dtype, np.floating):
             print(f"Warning: Casting data from {data.dtype} to float64.")
             data = data.astype(np.float64)

        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_parents)
        self._op = _op
        self.label = label

    def __repr__(self):
        data_str = f"array(shape={self.data.shape})" if self.data.ndim > 0 else f"scalar({self.data.item():.4f})"
        grad_str = f"array(shape={self.grad.shape})" if self.grad.ndim > 0 else f"scalar({self.grad.item():.4f})"
        return f"Value(data={data_str}, grad={grad_str}, op='{self._op}')"

    # --- Implement Core Operations ---
    ## Add operation with broadcasting has been given as an example 
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data + other.data
        out = Value(out_data, (self, other), '+')

        def _backward():
            grad_self = out.grad
            grad_other = out.grad
            
            if self.data.shape != grad_self.shape:
                axis_to_sum = tuple(range(grad_self.ndim - self.data.ndim))
                grad_self = np.sum(grad_self, axis=axis_to_sum)
                if grad_self.shape != self.data.shape:
                    dims_to_squeeze = tuple(i for i, dim in enumerate(self.data.shape) if dim == 1)
                    grad_self = np.sum(grad_self, axis=dims_to_squeeze, keepdims=True)

            if other.data.shape != grad_other.shape:
                axis_to_sum = tuple(range(grad_other.ndim - other.data.ndim))
                grad_other = np.sum(grad_other, axis=axis_to_sum)
                if grad_other.shape != other.data.shape:
                    dims_to_squeeze = tuple(i for i, dim in enumerate(other.data.shape) if dim == 1)
                    grad_other = np.sum(grad_other, axis=dims_to_squeeze, keepdims=True)
            
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __mul__(self, other):
        """ Multiplication operation. Handles broadcasting. """
        other = other if isinstance(other, Value) else Value(other)
        # --- Step 1 - Forward Pass ---
        out_data = self.data * other.data
        out = Value(out_data, (self, other), '*')

        # --- Step 2 - Define _backward for Multiplication ---
        def _backward():
            grad_self = out.grad * other.data
            grad_other = out.grad * self.data

            if self.data.shape != grad_self.shape:
                dims_to_sum = grad_self.ndim - self.data.ndim
                grad_self = np.sum(grad_self, axis=tuple(range(dims_to_sum)))
                keep_dims = tuple(i for i, dim in enumerate(self.data.shape) if dim == 1)
                if keep_dims:
                    grad_self = np.sum(grad_self, axis=keep_dims, keepdims=True)
            
            if other.data.shape != grad_other.shape:
                dims_to_sum = grad_other.ndim - other.data.ndim
                grad_other = np.sum(grad_other, axis=tuple(range(dims_to_sum)))
                keep_dims = tuple(i for i, dim in enumerate(other.data.shape) if dim == 1)
                if keep_dims:
                    grad_other = np.sum(grad_other, axis=keep_dims, keepdims=True)
            
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out
    
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other): return self * (other**-1)
    def __rtruediv__(self, other): return other * (self**-1)

    def __pow__(self, other):
        """ Power operation (only supports scalar power). """
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        # --- Step 1 - Forward Pass ---
        out_data = self.data ** other
        out = Value(out_data, (self,), f'**{other}')

        # --- Step 2 - Define _backward for Power ---
        def _backward():
            self.grad += (other * self.data**(other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """ Rectified Linear Unit (ReLU) activation. """
        # --- Step 1 - Forward Pass ---
        out_data = np.maximum(0, self.data)
        out = Value(out_data, (self,), 'ReLU')

        # --- Step 2 - Define _backward for ReLU ---
        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """ Exponential function. """
        # --- Step 1 - Forward Pass ---
        clipped_data = np.clip(self.data, -np.inf, 700)
        out_data = np.exp(clipped_data)
        out = Value(out_data, (self,), 'exp')

        # --- Step 2 - Define _backward for exp ---
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        """ Natural logarithm function (log base e). """
        # --- Step 1 - Forward Pass ---
        stable_data = np.maximum(self.data, 1e-15)
        out_data = np.log(stable_data)
        out = Value(out_data, (self,), 'log')

        # --- Step 2 - Define _backward for log ---
        def _backward():
            self.grad += (1 / stable_data) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """ Matrix multiplication (@ operator). """
        other = other if isinstance(other, Value) else Value(other)
        # --- Step 1 - Forward Pass ---
        out_data = self.data @ other.data
        out = Value(out_data, (self, other), '@')

        # --- Step 2 - Define _backward for matmul ---
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        """ Summation operation. """
        # --- Step 1 - Forward Pass ---
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Value(out_data, (self,), 'sum')

        # --- Step 2 - Define _backward for sum ---
        def _backward():
            grad_to_propagate = out.grad
            if not keepdims and axis is not None:
                grad_to_propagate = np.expand_dims(out.grad, axis)
            self.grad += np.ones_like(self.data) * grad_to_propagate

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """ Mean operation. """
        # --- Step 1 - Forward Pass ---
        # I'll implement this by reusing sum and division
        # to automatically handle the backward pass.
        if axis is None:
            N = self.data.size
        else:
            N = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[a] for a in axis])

        out = self.sum(axis=axis, keepdims=keepdims) * (1.0 / N)
        out._op = 'mean' # Just relabel the operation
        return out

    def backward(self):
        """ Performs backpropagation starting from this Value node. """
        # --- Step 1 - Topological Sort ---
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        
        build_topo(self)
        
        # --- Step 2 - Initialize Gradient ---
        self.grad = np.ones_like(self.data, dtype=np.float64)
        
        # --- Step 3 - Backward Pass ---
        for node in reversed(topo):
            node._backward()
