import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.array(v)
    
    if np.ndim(v)==1:
        v = np.expand_dims(v, axis = 0)
        
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)

    # Avoid division by zero for zero vectors
    # Where norm_v is zero, keep the original vector (which would be a zero vector)
    normalized_v = np.where(norm_v == 0, v, v / norm_v)

    if normalized_v.shape[0]==1:
        normalized_v = normalized_v[0]
        
    return normalized_v