import numpy as np


class GaussianBasisFeatures:
    def __init__(self, n_centers=100, sigma=5.0, random_state=None):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers_ = None
        self.random_state = random_state # For reproducibility if using random sampling


    def fit(self, X, y=None):
        """
        Select n_centers random points from X as RBF centers.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        rng = np.random.RandomState(self.random_state)
        n_centers = min(self.n_centers, X.shape[0])
        indices = rng.choice(X.shape[0], n_centers, replace=False)
        self.centers_ = X[indices]
        return self


    # def transform(self, X):
    #     """
    #     Transform input X into Gaussian RBF features.
    #     Output shape: (n_samples, n_centers)
    #     """


    #     if self.centers_ is None:
    #         raise RuntimeError("Transformer is not fitted yet.")
    #     # Apply the RBF formula: exp(-(||X - center||^2 / (2 * sigma^2)))
    
    
    #     # Compute squared Euclidean distance between each sample and each center
    #     # Using broadcasting: ||x - mu||^2 = (x - mu)^2 summed over features
    #     # X shape: (n_samples, n_features)
    #     # centers_ shape: (n_centers, n_features)
    #     # print("GaussianBasisFeatures: Transform method implementation note.")

    #     # Broadcasting to get (n_samples, n_centers, n_features)
    #     diff = X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]
    #     # Summing over the feature dimension (axis=2)
    #     sq_distances = np.sum(diff**2, axis=2)
        
    #     # Applying the RBF formula
    #     rbf_features = np.exp(-sq_distances / (2 * self.sigma**2))

    #     return rbf_features 


    def transform(self, X):
        """
        Transform input X into Gaussian RBF features. (Memory-Efficient Version) above commented code will cause 
        the memory error
        """
        if self.centers_ is None:
            raise RuntimeError("Transformer is not fitted yet.")

        n_samples = X.shape[0]
        # Create an empty array to store the new features
        rbf_features = np.zeros((n_samples, self.centers_.shape[0]))

        # Loop through each center to avoid large intermediate array 
        for i, center in enumerate(self.centers_):
            # Calculate squared Euclidean distance from all samples to this one center
            # This creates a vector of shape (n_samples,)
            sq_distances = np.sum((X - center)**2, axis=1)
            
            # Apply the RBF formula for this center and store it as a column
            rbf_features[:, i] = np.exp(-sq_distances / (2 * self.sigma**2))
            
        return rbf_features


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
