import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    
    
    T = scores.shape[-1]
    score_dim = np.ndim(scores)
    masked_scores = np.copy(scores)
    for i in range(T):
        if i+1<T:
            masked_scores[..., i, i+1:] = mask_value
        
        
    # pass
    return masked_scores