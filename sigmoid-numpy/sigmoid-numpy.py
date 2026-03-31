import numpy as np

def sigmoid(x):
# For x >= 0, exp(-x) is safe.
# For x < 0, computing exp(x) is safe because x is negative, so it becomes a small number instead of blowing up.
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)

    pos = x >= 0
    neg = ~pos

    out[pos] = 1 / (1 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1 + exp_x)

    return out