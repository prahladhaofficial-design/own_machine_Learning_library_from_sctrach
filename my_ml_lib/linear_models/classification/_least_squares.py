import numpy as np


class LeastSquaresClassifier:
    """
    Classifier using the least squares approach.


    Fits a linear model by minimizing the squared error between
    predictions and target labels (e.g., encoded as +1/-1 or one-hot).
    Prediction is typically based on the sign or argmax of the output.
    """
    def __init__(self):
        self.w_ = None # Weight vector (including bias)
        self.classes_ = None # To store the class labels


    def fit(self, X, y):
        """
        Fit the least squares classifier model.


        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels), shape (n_samples,).
        """
        # Implement the least squares fitting logic (from A1/if covered).
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # 1. Choose a target encoding T for the labels y (e.g., +1/-1 or one-hot).
        if n_classes == 2:
            # For binary case, using -1 and 1 for the two classes.
            T = np.where(y == self.classes_[0], -1, 1)
        else:
            # For multi-class, using one-hot encoding.
            T = np.zeros((n_samples, n_classes))
            for i, cls in enumerate(self.classes_):
                T[y == cls, i] = 1

        # 2. Augment X with a bias column.
        X_augmented = np.hstack([np.ones((n_samples, 1)), X])

        # 3. Solve for weights using the normal equation: w = (X^T X)^(-1) X^T T
        #    Or use np.linalg.pinv for stability: w = pinv(X) @ T
        self.w_ = np.linalg.pinv(X_augmented) @ T
        
        return self


    def predict(self, X):
        """
        Predict class labels for samples in X.


        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).


        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")


        # Implement prediction logic.
        # 1. Augment X with a bias column.
        X_augmented = np.hstack([np.ones((X.shape[0], 1)), X])

        # 2. Calculate scores: scores = X_augmented @ self.w_
        scores = X_augmented @ self.w_

        # 3. Determine predicted class based on scores (e.g., sign for binary, argmax for multi-class)
        if scores.ndim == 1 or scores.shape[1] == 1:
            # Binary case, prediction based on sign of the score.
            pred_indices = (scores.flatten() > 0).astype(int)
        else:
            # Multi-class case, prediction is the index of the max score.
            pred_indices = np.argmax(scores, axis=1)

        # 4. Map back to original class labels.
        predictions = self.classes_[pred_indices]

        return predictions
