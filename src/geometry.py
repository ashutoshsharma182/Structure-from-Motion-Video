"""Two-view geometry from the Computer Vision course exercise."""
import numpy as np

def normalize_points(x):
    """
    Args:
        x: 3xN arrays of N homogenous points in 2D

    Return:
        x_trans: 3xN matrix of transformed points
        T: the 3x3 transformation matrix, points_trans = T * points
    """
    # Shift origin to centroid
    x = x / x[2]
    center = np.mean(x, axis=1)
    T_center = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
    x_trans = T_center @ x

    # Normalize the distances
    scale = np.sqrt(2) / np.mean(np.linalg.norm(x_trans[:2], axis=0))
    T_scale = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    T = T_scale @ T_center
    x_trans = T @ x

    return x_trans, T


def get_fundamental_matrix(x0, x1):
    """
    Args:
        x0, x1: 3xN arrays of N homogenous points in 2D

    Returns:
        F: The 3x3 fundamental martix such that x0.T @ F @ x1 = 0
        e0: The epipole in image 0 such that F.T  @ e0 = 0
        e1: The epipole in image 1 such that F @ e1 = 0
    """
    x0 = x0 / x0[2]
    x1 = x1 / x1[2]

    # Constraint matrix
    n_pts = x0.shape[1]
    A = np.column_stack(
        (
            x0[0] * x1[0],
            x0[0] * x1[1],
            x0[0],
            x1[0] * x0[1],
            x0[1] * x1[1],
            x0[1],
            x1[0],
            x1[1],
            np.ones(n_pts),
        )
    )

    # SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank(F) = 2
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    # Epipoles
    e0 = U[:, -1] / U[-1, -1]
    e1 = Vt[-1, :] / Vt[-1, -1]

    return F, e0, e1


def get_fundamental_matrix_with_normalization(x0, x1):
    """
    Args:
        x0, x1: 3xN arrays of N homogenous points in 2D

    Returns:
        F: The 3x3 fundamental martix such that x0'*F*x1 = 0
        e0: The epipole in image 0 such that F'*e0 = 0
        e1: The epipole in image 1 such that F*e1 = 0
    """
    x0 = x0 / x0[2]
    x1 = x1 / x1[2]

    x0_norm, T0 = normalize_points(x0)
    x1_norm, T1 = normalize_points(x1)
    F_norm, _, _ = get_fundamental_matrix(x0_norm, x1_norm)

    # Undo the transformation
    F = T0.T @ F_norm @ T1

    # Compute epipoles
    U, _, Vt = np.linalg.svd(F)
    e0 = U[:, -1] / U[-1, -1]
    e1 = Vt[-1, :] / Vt[-1, -1]

    return F, e0, e1


def get_residual_distance(F, x0, x1):
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    # Epipolar lines
    l0 = F @ x1
    l1 = F.T @ x0

    # Normalize
    x0 = x0 / x0[2]
    x1 = x1 / x1[2]
    l0 = l0 / np.hypot(l0[0], l0[1])
    l1 = l1 / np.hypot(l1[0], l1[1])

    # Distance
    d0 = np.abs(np.sum(l0 * x0, axis=0))
    d1 = np.abs(np.sum(l1 * x1, axis=0))

    return d0, d1


def get_residual_error(F, x0, x1):
    d0, d1 = get_residual_distance(F, x0, x1)
    return 0.5 * (np.mean(d0) + np.mean(d1))


def get_inliers(F, x0, x1, eps):
    d0, d1 = get_residual_distance(F, x0, x1)
    indices = np.argwhere(np.logical_and(d0 < eps, d1 < eps))[..., 0]
    return indices


def get_fundamental_matrix_with_ransac(x0, x1, eps=10, n_iter=1000):
    """
    Args:
        x0, x1: 3xN arrays of N homogenoous points in 2D
        eps: Inlier threshold
        n_iter: Number of iterations

    Return:
        F: The 3x3 fundamental martix such taht x2'*F*x1 = 0
        e0: The epipole in image 1 such that F'*e0 = 0
        e1: The epipole in image 2 such that F*e1 = 0
        inlier_indices: Indices of inlier
    """
    n_pts = x1.shape[1]
    n_sample = 8

    indices = np.arange(n_pts)
    best_inlier_indices = []

    for i in range(n_iter):
        # Randomly select seed group of points
        sample_indices = np.random.choice(indices, size=n_sample, replace=False)
        x0_sample = x0[:, sample_indices]
        x1_sample = x1[:, sample_indices]

        # Compute transformation
        F, e0, e1 = get_fundamental_matrix_with_normalization(x0_sample, x1_sample)

        # Find inliers
        inlier_indices = get_inliers(F, x0, x1, eps)
        if len(inlier_indices) > len(best_inlier_indices):
            best_inlier_indices = inlier_indices

    if len(best_inlier_indices) < 8:
        raise RuntimeError("Too few inliers found.")

    # Estimate fundamental matrix again with all inliers
    F, e0, e1 = get_fundamental_matrix_with_normalization(
        x0[:, best_inlier_indices], x1[:, best_inlier_indices]
    )

    return F, e0, e1, best_inlier_indices


def vector_to_skew(vec):
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])


def triangulate(x0, x1, P0, P1):
    """Triangulate matching points.
    Args:
        x0, x1: 3xN matrices of matching points (homogeneous)
        P0, P1: 3X4 camera matrices

    Returns:
        X: 4xN matrix of points in world space
    """
    X = np.empty((4, x0.shape[1]), dtype=np.float64)
    for i in range(x0.shape[1]):
        A = np.concatenate(
            [vector_to_skew(x0[:, i])[:2] @ P0, vector_to_skew(x1[:, i])[:2] @ P1],
            axis=0,
        )
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        X[:, i] = Vt[-1, :] / Vt[-1, -1]
    return X


def get_reprojection_error(X, x, P):
    x_p = P @ X
    res = x - x_p / x_p[2, :]
    distance = np.hypot(res[0, :], res[1, :])
    return distance
