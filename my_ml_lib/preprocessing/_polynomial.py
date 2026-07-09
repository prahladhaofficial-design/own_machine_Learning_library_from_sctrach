import numpy as np
import itertools

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None

    def fit(self, X, y=None):
        self.n_input_features_ = X.shape[1]
        
        # Calculate the number of output features
        # This is a combinatorics problem: C(n + d, d)
        # where n is n_input_features_ and d is degree
        from math import comb
        self.n_output_features_ = comb(self.n_input_features_ + self.degree, self.degree)
        
        if not self.include_bias:
            # If no bias, subtract the 1 feature
            self.n_output_features_ -= 1
            
        return self

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def transform(self, X):
        if self.n_input_features_ is None:
            raise RuntimeError("Must fit before transforming.")

        n_samples = X.shape[0]
        
        # Generate combinations of feature indices
        # This determines which features to multiply for each new polynomial term
        feature_indices = range(self.n_input_features_)
        
        # Start with an empty list for the new features
        poly_features = []
        
        if self.include_bias:
            poly_features.append(np.ones((n_samples, 1)))

        # Generate terms for degree 1 up to self.degree
        for d in range(1, self.degree + 1):
            for indices in itertools.combinations_with_replacement(feature_indices, d):
                # Take the product of the features at the selected indices
                new_feature = np.prod(X[:, indices], axis=1, keepdims=True)
                poly_features.append(new_feature)

        # Concatenate all generated features
        return np.hstack(poly_features)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
