import numpy as np


class Perceptron:
    """
    Perceptron classifier.


    Simple algorithm for binary linear classification.
    Uses iterative updates based on misclassified points.
    """
    def __init__(self, learning_rate=0.01, max_iters=1000, random_state=None):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.random_state = random_state
        self.w_ = None # Weights (including bias)
        self.errors_ = [] # To store number of misclassifications per epoch


    def fit(self, X, y):
        """
        Fit perceptron model.


        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels, e.g., 0/1 or -1/1), shape (n_samples,).
        """
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        
        # 1. Initialize weights (e.g., randomly or to zeros). Include bias.
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=1 + n_features)
        
        # 2. Augment X with bias column.
        X_augmented = np.hstack([np.ones((n_samples, 1)), X])

        # 3. Ensure y labels are appropriate (+1/-1 often used).
        # (Assuming y is already in {-1, 1} format as suggested by the template)

        # Pocket algorithm setup
        best_w = self.w_.copy()
        best_errors = n_samples + 1
        self.errors_ = []

        # 4. Iterate up to max_iters:
        for _ in range(self.max_iters):
            current_errors = 0
            for i in range(n_samples):
                # Make a prediction
                score = np.dot(X_augmented[i], self.w_)
                prediction = 1 if score >= 0.0 else -1

                # If misclassified, update weights
                if y[i] != prediction:
                    self.w_ += self.learning_rate * y[i] * X_augmented[i]
                    current_errors += 1
            
            self.errors_.append(current_errors)
            
            # Pocket algorithm: keep track of the weights that achieved the lowest misclassification rate
            if current_errors < best_errors:
                best_errors = current_errors
                best_w = self.w_.copy()

            if best_errors == 0:
                break
        
        self.w_ = best_w

        return self


    def _predict_raw(self, X):
        """Calculate net input (scores)."""
        # Augment X
        X_augmented = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_augmented @ self.w_


    def predict(self, X):
        """
        Return class label after unit step.


        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).


        Returns:
            np.ndarray: Predicted class labels (e.g., 0/1 or -1/1 depending on fit).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")


        scores = self._predict_raw(X)
        predictions = np.where(scores >= 0.0, 1, -1) # Example for -1/1 labels
        
        return predictions
