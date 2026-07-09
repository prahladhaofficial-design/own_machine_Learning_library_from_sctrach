# my_ml_lib/nn/losses.py

import numpy as np
from .modules.base import Module
from .autograd import Value

### You would use this for one vs rest Logistic Regression
class BinaryCrossEntropyLoss(Module):
    """
    Computes the Binary Cross Entropy loss between logits and targets (0 or 1).
    Assumes the input is a single logit (pre-sigmoid score) per sample.
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def __call__(self, logits: Value, targets: np.ndarray) -> Value:
        """
        Calculates the Binary Cross Entropy loss.
        """
        targets_np = targets.astype(np.float64).reshape(-1, 1) if targets.ndim == 1 else targets.astype(np.float64)

        # --- Step 1 - Apply Sigmoid ---
        probs = (Value(1) + (-logits).exp())**-1

        # --- Step 2 - Numerical Stability (Clipping) ---
        # Stability is handled by Value.log() internally.
        probs_stable = probs

        # --- Step 3 - Wrap Targets ---
        targets_val = Value(targets_np)

        # --- Step 4 - Calculate BCE Formula ---
        loss_elements = -(targets_val * probs_stable.log() + (Value(1) - targets_val) * (Value(1) - probs_stable).log())

        # --- Step 5 - Apply Reduction ---
        if self.reduction == 'mean':
            loss = loss_elements.mean()
        elif self.reduction == 'sum':
            loss = loss_elements.sum()
        else: # 'none'
            loss = loss_elements

        return loss

    def __repr__(self):
        """String representation."""
        return f"BinaryCrossEntropyLoss(reduction='{self.reduction}')"

class CrossEntropyLoss(Module):
    """
    Computes the cross-entropy loss between input logits and target class indices.
    Combines LogSoftmax and NLLLoss.
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): Specifies the reduction: 'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction type: {reduction}. Must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def __call__(self, input_logits: Value, target: np.ndarray) -> Value:
        """
        Computes the cross-entropy loss using only Value operations.
        """
        batch_size, n_classes = input_logits.data.shape

        # --- Step 1 - Calculate LogSoftmax ---
        max_logits = np.max(input_logits.data, axis=1, keepdims=True)
        stable_logits = input_logits - max_logits
        exp_logits = stable_logits.exp()
        sum_exp_logits = exp_logits.sum(axis=1, keepdims=True)
        log_sum_exp = sum_exp_logits.log()
        log_probs = stable_logits - log_sum_exp

        # --- Step 2 - Create One-Hot Targets ---
        y_one_hot_np = np.zeros((batch_size, n_classes), dtype=np.float64)
        y_one_hot_np[np.arange(batch_size), target] = 1.0

        # --- Step 3 - Wrap One-Hot Targets ---
        y_one_hot_val = Value(y_one_hot_np)

        # --- Step 4 - Calculate NLL ---
        nll_per_sample = -(y_one_hot_val * log_probs).sum(axis=1)

        # --- Step 5 - Apply Reduction ---
        if self.reduction == 'mean':
            loss = nll_per_sample.mean()
        elif self.reduction == 'sum':
            loss = nll_per_sample.sum()
        else: # 'none'
            loss = nll_per_sample

        return loss

    def __repr__(self):
        """String representation."""
        return f"CrossEntropyLoss(reduction='{self.reduction}')"
