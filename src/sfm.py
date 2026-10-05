"""
Minimal sequential Structure-from-Motion.

    frames 0,1 : match -> F (RANSAC) -> E -> (R, t) -> triangulate        (two-view initialisation)
    frame k    : match with k-1 -> 2D-3D pairs -> PnP pose -> triangulate  (add one view at a time)
    all frames : bundle adjustment                                          (refine structure and motion)

Every new camera is localised against the EXISTING 3D points (PnP), so all cameras and points share one
coordinate frame and one scale. Chaining two-view poses instead would give each pair its own scale.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from geometry import get_fundamental_matrix_with_ransac, triangulate, get_reprojection_error


def detect(gray):
    """SIFT keypoints (Nx2 pixel coordinates) and their descriptors."""
    kps, des = cv2.SIFT_create(nfeatures=8000).detectAndCompute(gray, None)
    return np.array([kp.pt for kp in kps]), des


def match(des0, des1, ratio=0.75):
    """Nearest-neighbour descriptor matching with Lowe's ratio test. Returns Mx2 keypoint indices."""
    pairs = cv2.BFMatcher().knnMatch(des0, des1, k=2)
    return np.array([(m.queryIdx, m.trainIdx) for m, n in pairs if m.distance < ratio * n.distance])


def homogeneous(pts):
    return np.vstack([pts.T, np.ones(len(pts))])


def verified_matches(f0, f1, eps):
    """Descriptor matches between two frames, keeping only RANSAC inliers of the fundamental matrix."""
    m = match(f0['des'], f1['des'])
    F, _, _, inliers = get_fundamental_matrix_with_ransac(homogeneous(f0['kps'][m[:, 0]]),
                                                          homogeneous(f1['kps'][m[:, 1]]), eps=eps)
    return m[inliers], F


def add_points(f0, f1, m, P0, P1, points, colors, image, max_error):
    """Triangulate matches m between f0 and f1, keep sound points, and remember which keypoint made which point."""
    x0, x1 = homogeneous(f0['kps'][m[:, 0]]), homogeneous(f1['kps'][m[:, 1]])
    X = triangulate(x0, x1, P0, P1)
    error = np.maximum(get_reprojection_error(X, x0, P0), get_reprojection_error(X, x1, P1))
    in_front = ((P0 @ X)[2] > 0) & ((P1 @ X)[2] > 0)
    good = in_front & (error < max_error)

    for X_k, (k0, k1) in zip(X[:3, good].T, m[good]):
        f0['point'][k0] = f1['point'][k1] = len(points)
        points.append(X_k)
        u, v = np.round(f1['kps'][k1]).astype(int)
        colors.append(image[min(v, image.shape[0] - 1), min(u, image.shape[1] - 1), ::-1])  # BGR -> RGB
    return good.sum()


def bundle_adjust(poses, points, observations, K, log=print):
    """
    Bundle adjustment: refine all cameras P_i = K [R_i | t_i], all 3D points X_j and the focal length f
    by minimising the reprojection error  E(P, X) = sum_ij || x_ij - P_i X_j ||^2.
    observations: list of (camera i, point j, pixel x_ij). Each R_i is stored as a rotation vector (3 numbers).
    """
    cam = np.array([i for i, _, _ in observations])
    pt = np.array([j for _, j, _ in observations])
    x = np.array([xy for _, _, xy in observations])
    m, n = len(poses), len(points)

    def unpack(params):  # params = [f, (rvec_i, t_i) for every camera, X_j for every point]
        return params[0], params[1:1 + 6 * m].reshape(m, 6), params[1 + 6 * m:].reshape(n, 3)

    def residuals(params):
        f, cams, X = unpack(params)
        Xc = Rotation.from_rotvec(cams[cam, :3]).apply(X[pt]) + cams[cam, 3:]  # R_i X_j + t_i
        projected = f * Xc[:, :2] / Xc[:, 2:] + K[:2, 2]                       # K (R_i X_j + t_i), dehomogenised
        return (projected - x).ravel()

    # Each residual depends only on f, its own camera and its own point. Telling the solver this keeps it fast.
    sparsity = lil_matrix((2 * len(x), 1 + 6 * m + 3 * n), dtype=int)
    for k in range(2):
        rows = 2 * np.arange(len(x)) + k
        sparsity[rows, 0] = 1
        for c in range(6):
            sparsity[rows, 1 + 6 * cam + c] = 1
        for c in range(3):
            sparsity[rows, 1 + 6 * m + 3 * pt + c] = 1

    cams = [np.hstack([Rotation.from_matrix(R).as_rotvec(), t.ravel()]) for R, t in poses]
    params = np.hstack([K[0, 0], np.ravel(cams), points.ravel()])
    rmse = lambda p: np.sqrt(np.mean(residuals(p).reshape(-1, 2) ** 2) * 2)
    before = rmse(params)
    params = least_squares(residuals, params, jac_sparsity=sparsity, x_scale='jac').x
    log(f"Bundle adjustment: reprojection RMSE {before:.2f} px -> {rmse(params):.2f} px")

    f, cams, X = unpack(params)
    poses = [(Rotation.from_rotvec(c[:3]).as_matrix(), c[3:].reshape(3, 1)) for c in cams]
    return poses, X, f


def reconstruct(images, K, eps=1.5, max_error=4.0, log=print):
    """images: list of BGR images in capture order. Returns camera poses [(R, t)], points Nx3, colors Nx3, focal length."""
    np.random.seed(0)  # reproducible RANSAC
    frames = []
    for img in images:
        kps, des = detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        frames.append({'kps': kps, 'des': des, 'point': -np.ones(len(kps), dtype=int)})  # keypoint -> 3D point id
    points, colors = [], []

    # 1. Two-view initialisation from frames 0 and 1
    m, F = verified_matches(frames[0], frames[1], eps)
    E = K.T @ F.T @ K  # our F satisfies x0^T F x1 = 0; OpenCV expects x1^T E x0 = 0, hence F^T
    _, R, t, _ = cv2.recoverPose(E, frames[0]['kps'][m[:, 0]], frames[1]['kps'][m[:, 1]], K)  # cheirality picks 1 of 4
    poses = [(np.eye(3), np.zeros((3, 1))), (R, t)]
    P = [K @ np.hstack(p) for p in poses]
    n = add_points(frames[0], frames[1], m, P[0], P[1], points, colors, images[1], max_error)
    log(f"Frames 0-1: {len(m)} inlier matches -> {n} points")

    # 2. Add the remaining frames one by one
    for k in range(2, len(frames)):
        prev, cur = frames[k - 1], frames[k]
        m, _ = verified_matches(prev, cur, eps)

        # 2D-3D correspondences: keypoints of the previous frame that already have a 3D point
        known = prev['point'][m[:, 0]] >= 0
        obj = np.array([points[i] for i in prev['point'][m[known, 0]]])
        img = cur['kps'][m[known, 1]]
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=max_error) \
            if len(obj) >= 6 else (False, None, None, None)
        if not ok or inl is None or len(inl) < 20:
            log(f"Frame {k}: only {len(obj)} 2D-3D matches, cannot localise -> stopping here")
            break
        poses.append((cv2.Rodrigues(rvec)[0], tvec))
        P.append(K @ np.hstack(poses[-1]))
        cur['point'][m[known, 1][inl.ravel()]] = prev['point'][m[known, 0][inl.ravel()]]  # extend tracks

        n = add_points(prev, cur, m[~known], P[k - 1], P[k], points, colors, images[k], max_error)
        log(f"Frame {k}: PnP {len(inl)}/{len(obj)} inliers -> +{n} new points ({len(points)} total)")

    # 3. Bundle adjustment over every camera and point. Observations come from the keypoint -> 3D point tables.
    observations = [(i, j, frames[i]['kps'][k]) for i in range(len(poses))
                    for k, j in enumerate(frames[i]['point']) if j >= 0]
    poses, points, f = bundle_adjust(poses, np.array(points), observations, K, log)
    return poses, points, np.array(colors, dtype=np.uint8), f
