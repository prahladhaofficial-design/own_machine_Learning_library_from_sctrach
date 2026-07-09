# my_ml_lib/linear_models/classification/_logistic.py

import numpy as np

class LogisticRegression:
    """
    L2-Regularized Logistic Regression classifier using Stochastic Gradient Descent (SGD).
    """

    def __init__(self, alpha=0.01, epochs=50, lr=0.01, batch_size=64,
                 fit_intercept=True, random_state=None):
        """
        Args:
            alpha (float): L2 regularization strength.
            epochs (int): Number of passes over the training data.
            lr (float): The learning rate for SGD.
            batch_size (int): The number of samples per gradient update.
            fit_intercept (bool): Whether to add a bias term (intercept).
            random_state (int, optional): Seed for shuffling reproducibility.
        """
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.w_ = None  # Learned weights

    def _sigmoid(self, z):
        """Numerically stable sigmoid function."""
        # Clip z to avoid overflow/underflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X, y):
        """
        Fit the L2-regularized logistic regression model using mini-batch SGD.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values (0 or 1), shape (n_samples,).
        """
        n_samples, n_features = X.shape

        # --- 1. Add intercept term (bias) if required ---
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((n_samples, 1)), X])
            self.w_ = np.zeros(n_features + 1)
        else:
            X_aug = X
            self.w_ = np.zeros(n_features)
            
        # --- 2. SGD Training Loop ---
        # Set seed for reproducibility of shuffling
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        for epoch in range(self.epochs):
            # Shuffle the data at the beginning of each epoch
            shuffled_indices = np.random.permutation(n_samples)
            X_shuffled = X_aug[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            
            # Iterate over mini-batches
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                # --- 2a. Calculate predictions (h) for the batch ---
                h = self._sigmoid(X_batch @ self.w_)
                
                # --- 2b. Calculate gradient of the loss ---
                # Gradient of Binary Cross-Entropy + L2 Regularization
                gradient = (1 / y_batch.size) * X_batch.T @ (h - y_batch)
                
                # Add regularization term (but don't regularize the intercept/bias)
                if self.fit_intercept:
                    gradient[1:] += self.alpha * self.w_[1:]
                else:
                    gradient += self.alpha * self.w_
                
                # --- 2c. Update weights ---
                self.w_ -= self.lr * gradient
                
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Returns:
            np.ndarray: Predicted probabilities [P(y=0|X), P(y=1|X)].
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")

        if self.fit_intercept:
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_aug = X
            
        prob_y1 = self._sigmoid(X_aug @ self.w_)
        prob_y0 = 1 - prob_y1
        return np.column_stack([prob_y0, prob_y1])

    def predict(self, X):
        """
        Predict class labels (0 or 1) for samples in X.
        """
        # Get the probability of the positive class (y=1)
        prob_y1 = self.predict_proba(X)[:, 1]
        return (prob_y1 >= 0.5).astype(int)

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)
