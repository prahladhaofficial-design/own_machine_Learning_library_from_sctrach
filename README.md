# From-Scratch ML Library for Fashion-MNIST Experiments

A compact machine learning library with preprocessing, model selection, classical models, naive Bayes, and a minimal autograd-based neural network stack.

## Overview

This repository is built like a small ML package rather than a single experiment. It implements reusable components that mirror common machine learning library abstractions.
The project demonstrates how models, preprocessing, validation, and neural-network primitives can be organized into a coherent local library.

## Project Design

- Separate datasets, preprocessing, model selection, linear models, naive Bayes, and neural-network components into package modules.
- Implement classical methods from scratch to make assumptions and update rules visible.
- Use scripts and notebooks to compare models on spam, Fashion-MNIST-style, and synthetic data.
- Save selected artifacts and graph visualizations for review.

## Key Capabilities

- Dataset loaders for Spambase, Fashion-MNIST, and synthetic noisy sine data.
- Train-test split and K-fold cross-validation helpers.
- Standard scaling, polynomial features, and Gaussian basis features.
- Bernoulli and Gaussian naive Bayes classifiers.
- LDA, least-squares classification, logistic regression, perceptron, Bayesian regression, and ridge regression.
- Minimal scalar autograd engine with neural-network modules, activations, losses, and SGD.
- Saved best-model artifact and MLP computation-graph visualization.

## Repository Structure

```text
my_ml_lib/datasets/         Dataset loading and synthetic data generation.
my_ml_lib/preprocessing/    Scaling and feature transforms.
my_ml_lib/model_selection/  Splitting and validation helpers.
my_ml_lib/linear_models/    Linear regression and classification models.
my_ml_lib/naive_bayes/      Naive Bayes models.
my_ml_lib/nn/               Autograd and neural-network components.
capstore_showdown.ipynb     Model-comparison notebook.
run_spam_experiment.py      Spam experiment runner.
create_best_model.py        Saved-model generation script.
```

## Running Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_spam_experiment.py
python create_best_model.py
jupyter notebook capstore_showdown.ipynb
```

## Results and Artifacts

- `saved_models/best_model.npz` stores the selected model artifact.
- `mlp_computation_graph.png` visualizes the neural computation graph.
- `docs/model-showdown-report.md` summarizes model-comparison results.
- Reusable source modules live under `my_ml_lib/`.

## Reproducibility Notes

- Run the script-based checks first to validate package imports before opening experiment notebooks.
- Keep local dataset files in the expected locations when using non-synthetic loaders.
- Regenerate saved_models/best_model.npz after changing model-selection logic.
- Use fixed seeds and identical folds when comparing estimators in the model-selection utilities.

## Detailed Workflow

- Start by reading the `my_ml_lib/` package layout to understand available components.
- Run `run_spam_experiment.py` to test classical classification workflows.
- Run `create_best_model.py` to regenerate the saved model artifact.
- Open `capstore_showdown.ipynb` to inspect model comparisons and experiment reasoning.
- Use `mlp_computation_graph.png` to understand how the neural computation graph is represented.

## Implementation Details

- The library is organized into datasets, preprocessing, model selection, linear models, naive Bayes, and neural-network modules.
- The neural-network stack includes scalar autograd, layers, activations, losses, and optimization.
- Classical models are implemented as reusable components rather than one-off notebook cells.
- The saved model file demonstrates a complete train-and-export workflow.
- The package structure makes the project easier to extend with additional estimators.

## Data and Inputs

- Dataset loaders support Spambase, Fashion-MNIST-style data, and synthetic noisy sine data.
- Some datasets may need to be available locally depending on the experiment being run.
- Synthetic data utilities are included for controlled regression experiments.
- Saved model artifacts are stored separately from library source code.

## How To Interpret the Results

- Classical models provide baselines before neural methods are considered.
- Cross-validation utilities help compare models more fairly than a single split.
- Autograd visualization helps inspect how scalar operations compose into a graph.
- The notebook should be used to understand why one model is selected over another.

## Limitations and Assumptions

- The library is educational and compact; it is not intended to replace mature ML frameworks.
- The autograd engine is minimal and optimized for clarity rather than speed.
- Some experiments may require dataset files that are not generated automatically.
- The model APIs can be further standardized across estimators.


## Review Guide

Use this checklist when reviewing the repository on GitHub:

- Start with the overview to understand the project objective before opening notebooks or source files.
- Read the repository structure section to identify the main entry point.
- Inspect generated outputs before rerunning experiments; they provide a quick view of the current project state.
- Review scripts and notebooks together when both are present, since notebooks often explain decisions while scripts preserve reusable execution paths.
- Check the limitations section before interpreting results too strongly.
- Use the possible extensions section to understand how the project can grow without changing its core purpose.

## Troubleshooting

Common issues and fixes:

- If imports fail, confirm that the virtual environment is activated and dependencies were installed from `requirements.txt`.
- If a notebook cell fails because a variable is missing, restart the kernel and run the notebook from the first cell.
- If a script cannot find a data file, run it from the repository root unless the README gives a different working directory.
- If plots are not regenerated, check that the output folder exists and that the script has write permission.
- If a package version causes errors, create a fresh environment and reinstall dependencies before debugging the source code.
- If GPU-related packages are unavailable, run CPU-compatible sections first and treat heavier deep-learning experiments separately.

## Development Notes

The repository is organized to be understandable before it is optimized. Most code favors clear structure, visible intermediate steps, and saved artifacts over highly compressed implementations. When extending the project, prefer small changes that keep the original experiment reproducible.

Recommended development practices:

- Keep new data files under `data/`, `datasets/`, or the existing local data folder for that project.
- Keep generated figures under `figures/`, `plots/`, `outputs/`, or `results/` depending on the existing convention.
- Add new experiments as separate scripts or clearly labeled notebook sections.
- Avoid overwriting existing outputs unless the experiment is intentionally being refreshed.
- Document any new dependency in `requirements.txt`.
- Prefer relative paths so the repository works after cloning on another machine.

## Project Status

This repository is prepared as a standalone GitHub project. The current version includes the core implementation, documentation, and available artifacts needed to understand the work. Future improvements can focus on stronger automation, more tests, richer visual summaries, and cleaner packaging where appropriate.

## Possible Extensions

- Add unit tests for each estimator and transformer.
- Introduce a common `fit`, `predict`, and `score` interface across all models.
- Add model serialization utilities for every estimator.
- Publish the package locally with a minimal `pyproject.toml`.

## Tech Stack

Python, NumPy, Pandas, scikit-image, Matplotlib, tqdm, Jupyter Notebook.

## Author

Prahladha, IIT Hyderabad
