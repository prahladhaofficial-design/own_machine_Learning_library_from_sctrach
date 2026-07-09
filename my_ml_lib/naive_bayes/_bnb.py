import numpy as np

class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes (BNB) classifier.

    Assumes features are binary (0 or 1), conditionally independent
    given the class, and follow a Bernoulli distribution within each class.
    Often used for text classification with presence/absence features.
    """
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Additive (Laplace/Lidstone) smoothing parameter
                           (0 for no smoothing). Corresponds to Beta(alpha, alpha) prior.
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None # Log P(y=k)
        self.feature_log_prob_ = None # Log P(x_j=1 | y=k), shape (n_classes, n_features)
        # We implicitly store log P(x_j=0 | y=k) as log(1 - exp(log P(x_j=1 | y=k)))

    def fit(self, X, y):
        """
        Fit Bernoulli Naive Bayes according to X, y.

        Assumes X contains binary features (0 or 1).

        Args:
            X (np.ndarray): Training vectors (binary), shape (n_samples, n_features).
            y (np.ndarray): Target values (class labels), shape (n_samples,).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        X_binary = (X > 0).astype(float)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for idx, cls in enumerate(self.classes_):
            X_cls = X_binary[y == cls]
            class_count = X_cls.shape[0]
            self.class_log_prior_[idx] = np.log(class_count / X_binary.shape[0])
            smoothed_count = X_cls.sum(axis=0) + self.alpha
            smoothed_total = class_count + 2.0 * self.alpha
            feature_prob = smoothed_count / smoothed_total
            feature_prob = np.clip(feature_prob, 1e-12, 1.0 - 1e-12)
            self.feature_log_prob_[idx] = np.log(feature_prob)

        return self

    def predict_log_proba(self, X):
        """
        Calculate log probability estimates for samples in X.

        Assumes X contains binary features (0 or 1).

        Args:
            X (np.ndarray): Samples (binary), shape (n_samples, n_features).

        Returns:
            np.ndarray: Log probability of samples for each class, shape (n_samples, n_classes).
        """
        if self.classes_ is None:
             raise RuntimeError("Model is not fitted yet.")

        X_binary = (np.asarray(X) > 0).astype(float)
        log_prob_x1 = self.feature_log_prob_
        prob_x1 = np.exp(log_prob_x1)
        log_prob_x0 = np.log1p(-prob_x1)

        joint_log_likelihood = np.zeros((X_binary.shape[0], len(self.classes_)))
        for idx, _ in enumerate(self.classes_):
            log_likelihood = (
                X_binary * log_prob_x1[idx] + (1.0 - X_binary) * log_prob_x0[idx]
            ).sum(axis=1)
            joint_log_likelihood[:, idx] = self.class_log_prior_[idx] + log_likelihood
        return joint_log_likelihood

    def predict_proba(self, X):
        """
        Calculate probability estimates for samples in X.

        Args:
            X (np.ndarray): Samples (binary), shape (n_samples, n_features).

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
            X (np.ndarray): Samples (binary), shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
