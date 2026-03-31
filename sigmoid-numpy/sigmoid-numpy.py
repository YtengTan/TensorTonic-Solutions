import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    # pass
    x = np.array(x)
    sig_val = 1 / (1+np.exp(-x))
    return sig_val