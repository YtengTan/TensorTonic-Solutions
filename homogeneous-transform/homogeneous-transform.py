import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T = np.array(T)
    points = np.array(points)
    dim_1 = False
    if np.ndim(points)==1:
        dim_1 = True
        points = np.expand_dims(points, axis=0)
    
    

    column_ones = np.ones(points.shape[0])
    column_ones = np.expand_dims(column_ones, axis=1)
    
    homo_points = np.hstack((points, column_ones))

    homo_points_transformed = T @ homo_points.T

    homo_points_transformed = homo_points_transformed.T
    points_transformed = homo_points_transformed[:, :-1]

    if points_transformed.shape[0]==1:
        points_transformed = points_transformed[0]
    
    return points_transformed