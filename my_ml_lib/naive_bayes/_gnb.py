import numpy as np

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes (GNB) classifier.

    Assumes features are conditionally independent given the class,
    and each feature follows a Gaussian distribution within each class.
    """
    def __init__(self):
        self.classes_ = None
        self.class_priors_ = None # P(y=k)
        self.theta_ = None # Mean of each feature per class, shape (n_classes, n_features)
        self.var_ = None   # Variance of each feature per class, shape (n_classes, n_features)
        self.epsilon_ = 1e-9 # To prevent division by zero in variance

    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes according to X, y.

        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels), shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.class_priors_ = np.zeros(n_classes)
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))

        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.class_priors_[idx] = X_cls.shape[0] / X.shape[0]
            self.theta_[idx] = X_cls.mean(axis=0)
            self.var_[idx] = X_cls.var(axis=0) + self.epsilon_

        return self

    def _gaussian_log_pdf(self, X, class_idx):
        """Calculate log probability density function for Gaussian."""
        mean = self.theta_[class_idx]
        var = self.var_[class_idx]
        log_scale = -0.5 * np.sum(np.log(2.0 * np.pi * var))
        quadratic = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
        return log_scale + quadratic


    def predict_log_proba(self, X):
        """
        Calculate log probability estimates for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Log probability of samples for each class, shape (n_samples, n_classes).
        """
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet.")

        X = np.asarray(X, dtype=float)
        joint_log_likelihood = np.zeros((X.shape[0], len(self.classes_)))
        for idx, _ in enumerate(self.classes_):
            class_log_prior = np.log(self.class_priors_[idx])
            joint_log_likelihood[:, idx] = (
                class_log_prior + self._gaussian_log_pdf(X, idx)
            )
        return joint_log_likelihood

    def predict_proba(self, X):
        """
        Calculate probability estimates for samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability of samples for each class, shape (n_samples, n_classes).
        """
        log_proba = self.predict_log_proba(X)
        max_log_proba = np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba - max_log_proba)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba


    def predict(self, X):
        """
        Perform classification on samples in X.

        Args:
            X (np.ndarray): Samples, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
