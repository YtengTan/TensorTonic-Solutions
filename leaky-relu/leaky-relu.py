import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    # pass
    x_array = np.array(x).astype(float)

    smaller_than_zero_mask = x_array<0.0
        
    x_array[smaller_than_zero_mask] = x_array[smaller_than_zero_mask]*alpha

    return x_array