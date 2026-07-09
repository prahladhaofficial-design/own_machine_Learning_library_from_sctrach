# In create_best_model.py

from my_ml_lib.nn.modules.linear import Linear
from my_ml_lib.nn.modules.containers import Sequential
from my_ml_lib.nn.modules.activations import ReLU

def initialize_best_model():
    """
    This function returns an instance of the best model's architecture.
    It now returns a Sequential model directly to match the saved weights.
    """
    # This architecture MUST match your best trained model
    model = Sequential(
        Linear(in_features=784, out_features=128),
        ReLU(),
        Linear(in_features=128, out_features=10)
    )
    return model


# from my_ml_lib.nn.modules.base import Module
# from my_ml_lib.nn.modules.linear import Linear
# from my_ml_lib.nn.modules.containers import Sequential
# from my_ml_lib.nn.modules.activations import ReLU

# class MyMLP(Module):
#     def __init__(self, n_features, n_classes, hidden_size):
#         super().__init__()
#         self.network = Sequential(
#             Linear(n_features, hidden_size),
#             ReLU(),
#             Linear(hidden_size, n_classes)
#         )
#     def __call__(self, X):
#         return self.network(X)
#     def __repr__(self):
#         return f"{self.__class__.__name__}(\n{repr(self.network)}\n)"

# def initialize_best_model():
#     return MyMLP(n_features=784, n_classes=10, hidden_size=128)
