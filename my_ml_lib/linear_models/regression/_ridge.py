import numpy as np


class RidgeRegression:
    """
    Linear least squares with L2 regularization (Ridge Regression).


    Minimizes objective function: ||y - Xw||^2 + alpha * ||w||^2
    """
    def __init__(self, alpha=1.0, fit_intercept=True):
        """
        Args:
            alpha (float): Regularization strength; must be a positive float.
                           Larger values specify stronger regularization.
            fit_intercept (bool): Whether to calculate the intercept for this model.
                                  If set to False, no intercept will be used.
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None # Weights for the features
        self.intercept_ = None # Intercept (bias term)


    def fit(self, X, y):
        """
        Fit Ridge regression model.


        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values, shape (n_samples,).
        """
        # Implement Ridge Regression fitting using the normal equation.
        n_samples, n_features = X.shape

        # 1. Handle fit_intercept: If True, augment X with a column of ones.
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((n_samples, 1)), X])
            num_params = n_features + 1
        else:
            X_aug = X
            num_params = n_features
            
        # 2. Create the identity matrix I.
        # 3. Important: Do not regularize the intercept term.
        I = np.eye(num_params)
        if self.fit_intercept:
            I[0, 0] = 0 # Set top-left element to 0
        
        # 4. Solve the normal equation: w = (X^T X + alpha*I)^(-1) X^T y
        # Using np.linalg.solve(A, b) is better than inv(A) @ b
        A = X_aug.T @ X_aug + self.alpha * I
        b = X_aug.T @ y
        
        weights = np.linalg.solve(A, b)

        
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights
            self.intercept_ = 0.0
            
        return self


    def predict(self, X):
        """
        Predict using the linear model.


        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).


        Returns:
            np.ndarray: Predicted values, shape (n_samples,).
        """
        if self.coef_ is None:
              raise RuntimeError("Model is not fitted yet.")


        # Implement prediction: y_pred = X @ coef_ + intercept_
        y_pred = X @ self.coef_ + self.intercept_
        
        return y_pred
