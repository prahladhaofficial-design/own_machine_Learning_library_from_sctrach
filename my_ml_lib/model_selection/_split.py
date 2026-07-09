# my_ml_lib/model_selection/_split.py
import numpy as np


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Split X and y arrays into random train and test subsets.


    Args:
        X (array-like): Feature data, shape (n_samples, n_features).
        y (array-like): Target labels, shape (n_samples,).
        test_size (float or int): Proportion (0.0-1.0) or absolute number for the test split.
        shuffle (bool): Whether to shuffle data before splitting.
        random_state (int, optional): Seed for shuffling reproducibility.


    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    n_samples = X.shape[0]
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")


    # --- Step 1 - Calculate n_test and n_train ---
    if isinstance(test_size, float) and 0.0 < test_size < 1.0:
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int) and 0 < test_size < n_samples:
        n_test = test_size
    else:
        raise ValueError("test_size must be a float between 0.0 and 1.0 or an int.")
    
    n_train = n_samples - n_test


    # --- Step 2 - Create and Shuffle Indices ---
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)


    # --- Step 3 - Split Indices ---
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]


    # --- Step 4 - Split Arrays ---
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]


    return X_train, X_test, y_train, y_test


def train_test_val_split(X, y,
                         train_size=0.7,
                         val_size=0.15,
                         test_size=0.15,
                         shuffle=True,
                         random_state=None):
    """
    Split X and y arrays into random train, validation, and test subsets.


    Args:
        X (array-like): Feature data, shape (n_samples, n_features).
        y (array-like): Target labels, shape (n_samples,).
        train_size (float): Proportion for the train split (0.0 to 1.0).
        val_size (float): Proportion for the validation split (0.0 to 1.0).
        test_size (float): Proportion for the test split (0.0 to 1.0). Must sum to 1.0.
        shuffle (bool): Whether to shuffle data before splitting.
        random_state (int, optional): Seed for shuffling reproducibility.


    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n_samples = X.shape[0]
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")


    # --- Step 1 - Validate Proportions ---
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Proportions for train, val, and test splits must sum to 1.0.")


    # --- Step 2 - Calculate Split Sizes ---
    n_train = int(np.floor(n_samples * train_size))
    n_val = int(np.floor(n_samples * val_size))
    # Assign the rest to test set to ensure all data is used
    n_test = n_samples - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError("One of the splits resulted in 0 samples. Adjust proportions.")


    # --- Step 3 - Create and Shuffle Indices ---
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)


    # --- Step 4 - Split Indices ---
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val:]


    # --- Step 5 - Split Arrays ---
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test
