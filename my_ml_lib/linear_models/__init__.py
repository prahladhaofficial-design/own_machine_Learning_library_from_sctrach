# my_ml_lib/linear_models/__init__.py

# Make the classification and regression modules available when linear_models is imported
from . import classification
from . import regression

# This list tells Python what to import when a user does 'from . import *'
__all__ = [
    'classification',
    'regression'
]
