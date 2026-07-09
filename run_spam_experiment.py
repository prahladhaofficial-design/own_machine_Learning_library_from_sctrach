# run_spam_experiment_sgd.py

import numpy as np
import os
import itertools

# --- Boilerplate: Imports ---
try:
    from my_ml_lib.datasets import load_spambase, DatasetNotFoundError
    from my_ml_lib.preprocessing import StandardScaler
    from my_ml_lib.linear_models.classification import LogisticRegression # This is now your SGD version
    from my_ml_lib.model_selection import KFold, train_test_split
except ImportError as e:
    print(f"Error importing library components: {e}")
    print("Please ensure your my_ml_lib structure and __init__.py files are correct.")
    exit()
# --- End Boilerplate ---


# --- Path Configuration ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # This fallback is for running in an interactive environment like a notebook
    SCRIPT_DIR = os.getcwd() 
DATA_FOLDER = os.path.join(SCRIPT_DIR, "data")
# --- End Path Configuration ---


# --- Experiment Constants ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS_CV = 5
# NEW here : Hyperparameter grid for SGD-based Logistic Regression
PARAM_GRID = {
    'lr': [0.01, 0.001],
    'epochs': [200, 500],
    'alpha': [0.01, 0.1, 1.0]
}
# --- End Constants ---


# --- Step 1 - Load Data ---
try:
    X, y = load_spambase(data_folder=DATA_FOLDER)
    print(f"Successfully loaded Spambase dataset. X shape: {X.shape}, y shape: {y.shape}")
except DatasetNotFoundError:
    print(f"Error: 'spambase.data' not found in the expected directory: {DATA_FOLDER}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()


# --- Step 2 - Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
)
print(f"Data split into {len(X_train)} training and {len(X_test)} test samples.")


# --- Helper Function for Cross-Validation (Updated for SGD) ---
def find_best_params_sgd(X_train_cv, y_train_cv, param_grid, n_splits, random_state):
    """Performs K-Fold CV to find the best hyperparameters for the SGD Logistic Regression."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_params_found = None
    best_avg_accuracy = 0.0

    # Create all combinations of parameters from the grid
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    print(f"Running {n_splits}-fold CV for {len(all_params)} parameter combinations...")
    for params in all_params:
        fold_accuracies = []
        for train_idx, val_idx in kf.split(X_train_cv):
            X_fold_train, X_fold_val = X_train_cv[train_idx], X_train_cv[val_idx]
            y_fold_train, y_fold_val = y_train_cv[train_idx], y_train_cv[val_idx]
            
            # Instantiate model with the current set of parameters
            model = LogisticRegression(**params, random_state=random_state)
            model.fit(X_fold_train, y_fold_train)
            accuracy = model.score(X_fold_val, y_fold_val)
            fold_accuracies.append(accuracy)
        
        avg_accuracy = np.mean(fold_accuracies)
        print(f"  - Params: {params} -> Avg. CV Accuracy: {avg_accuracy:.4f}")

        if avg_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_accuracy
            best_params_found = params
            
    print(f"-> Best params found: {best_params_found} (CV Accuracy: {best_avg_accuracy:.4f})")
    return best_params_found


# --- Experiment with RAW Data ---
print("\n--- Experiment: Raw Data ---")
best_params_raw = find_best_params_sgd(X_train, y_train, PARAM_GRID, N_SPLITS_CV, RANDOM_STATE)
final_model_raw = LogisticRegression(**best_params_raw, random_state=RANDOM_STATE)
final_model_raw.fit(X_train, y_train)
train_error_raw = 1 - final_model_raw.score(X_train, y_train)
test_error_raw = 1 - final_model_raw.score(X_test, y_test)


# --- Experiment with STANDARDIZED Data ---
print("\n--- Experiment: Standardized Data ---")
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
best_params_std = find_best_params_sgd(X_train_std, y_train, PARAM_GRID, N_SPLITS_CV, RANDOM_STATE)
final_model_std = LogisticRegression(**best_params_std, random_state=RANDOM_STATE)
final_model_std.fit(X_train_std, y_train)
train_error_std = 1 - final_model_std.score(X_train_std, y_train)
test_error_std = 1 - final_model_std.score(X_test_std, y_test)


# --- Report Results ---
print("\n--- Summary Results (SGD-based Logistic Regression) ---")
print(f"{'Preprocessing':<15} | {'Best Params':<40} | {'Train Error':<15} | {'Test Error':<15}")
print(f":----------------|-----------------------------------------:|----------------:|---------------:")
print(f"{'Raw':<15} | {str(best_params_raw):<40} | {train_error_raw:<15.4f} | {test_error_raw:<15.4f}")
print(f"{'Standardized':<15} | {str(best_params_std):<40} | {train_error_std:<15.4f} | {test_error_std:<15.4f}")

'''
 The output when this code cell runs: 

Successfully loaded Spambase dataset. X shape: (4601, 57), y shape: (4601,)
Data split into 3681 training and 920 test samples.

--- Experiment: Raw Data ---
Running 5-fold CV for 12 parameter combinations...
  - Params: {'lr': 0.01, 'epochs': 200, 'alpha': 0.01} -> Avg. CV Accuracy: 0.7004
  - Params: {'lr': 0.01, 'epochs': 200, 'alpha': 0.1} -> Avg. CV Accuracy: 0.6417
  - Params: {'lr': 0.01, 'epochs': 200, 'alpha': 1.0} -> Avg. CV Accuracy: 0.6327
  - Params: {'lr': 0.01, 'epochs': 500, 'alpha': 0.01} -> Avg. CV Accuracy: 0.7207
  - Params: {'lr': 0.01, 'epochs': 500, 'alpha': 0.1} -> Avg. CV Accuracy: 0.6653
  - Params: {'lr': 0.01, 'epochs': 500, 'alpha': 1.0} -> Avg. CV Accuracy: 0.6387
  - Params: {'lr': 0.001, 'epochs': 200, 'alpha': 0.01} -> Avg. CV Accuracy: 0.6895
  - Params: {'lr': 0.001, 'epochs': 200, 'alpha': 0.1} -> Avg. CV Accuracy: 0.7042
  - Params: {'lr': 0.001, 'epochs': 200, 'alpha': 1.0} -> Avg. CV Accuracy: 0.6262
  - Params: {'lr': 0.001, 'epochs': 500, 'alpha': 0.01} -> Avg. CV Accuracy: 0.7308
  - Params: {'lr': 0.001, 'epochs': 500, 'alpha': 0.1} -> Avg. CV Accuracy: 0.7129
  - Params: {'lr': 0.001, 'epochs': 500, 'alpha': 1.0} -> Avg. CV Accuracy: 0.6678
-> Best params found: {'lr': 0.001, 'epochs': 500, 'alpha': 0.01} (CV Accuracy: 0.7308)

--- Experiment: Standardized Data ---
Running 5-fold CV for 12 parameter combinations...
  - Params: {'lr': 0.01, 'epochs': 200, 'alpha': 0.01} -> Avg. CV Accuracy: 0.9161
  - Params: {'lr': 0.01, 'epochs': 200, 'alpha': 0.1} -> Avg. CV Accuracy: 0.8992
  - Params: {'lr': 0.01, 'epochs': 200, 'alpha': 1.0} -> Avg. CV Accuracy: 0.8104
  - Params: {'lr': 0.01, 'epochs': 500, 'alpha': 0.01} -> Avg. CV Accuracy: 0.9174
  - Params: {'lr': 0.01, 'epochs': 500, 'alpha': 0.1} -> Avg. CV Accuracy: 0.8995
  - Params: {'lr': 0.01, 'epochs': 500, 'alpha': 1.0} -> Avg. CV Accuracy: 0.8112
  - Params: {'lr': 0.001, 'epochs': 200, 'alpha': 0.01} -> Avg. CV Accuracy: 0.9090
  - Params: {'lr': 0.001, 'epochs': 200, 'alpha': 0.1} -> Avg. CV Accuracy: 0.9006
  - Params: {'lr': 0.001, 'epochs': 200, 'alpha': 1.0} -> Avg. CV Accuracy: 0.8291
  - Params: {'lr': 0.001, 'epochs': 500, 'alpha': 0.01} -> Avg. CV Accuracy: 0.9117
  - Params: {'lr': 0.001, 'epochs': 500, 'alpha': 0.1} -> Avg. CV Accuracy: 0.8995
  - Params: {'lr': 0.001, 'epochs': 500, 'alpha': 1.0} -> Avg. CV Accuracy: 0.8142
-> Best params found: {'lr': 0.01, 'epochs': 500, 'alpha': 0.01} (CV Accuracy: 0.9174)

--- Summary Results (SGD-based Logistic Regression) ---
Preprocessing   | Best Params                              | Train Error     | Test Error
:----------------|-----------------------------------------:|----------------:|---------------:
Raw             | {'lr': 0.001, 'epochs': 500, 'alpha': 0.01} | 0.1432          | 0.1391
Standardized    | {'lr': 0.01, 'epochs': 500, 'alpha': 0.01} | 0.0799          | 0.0902


'''
