import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.array(x)
    soft_max_val = 0.

    if np.ndim(x) == 1:
        x_shifted = x - np.max(x)
        denominator = np.exp(x_shifted).sum()
        soft_max_val = np.exp(x_shifted) / denominator

    elif np.ndim(x) == 2:
        # Subtract the maximum value from each row for numerical stability
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        denominators = np.exp(x_shifted).sum(axis=1, keepdims=True)
        soft_max_val = np.exp(x_shifted) / denominators

    return soft_max_val