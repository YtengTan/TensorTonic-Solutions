def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    X_array = np.array(X)
    W_array = np.array(W)
    b_array = np.array(b)

    return (X_array@W_array+b_array).tolist()