"""Outputs: coloured PLY and a 360-degree video rendered with a virtual pinhole camera."""
import cv2
import imageio
import numpy as np


def save_ply(path, points, colors, poses):
    """
    Saved in a viewer-friendly frame: centred at the origin, y up, -z the average viewing direction
    (MeshLab/CloudCompare convention), with far-away stray points dropped so 'zoom to fit' frames the scene.
    """
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    keep = dist < 2 * np.percentile(dist, 90)
    up, forward = scene_axes(poses)
    R = np.stack([np.cross(forward, up), up, -forward])  # rows: right, up, back
    points, colors = (points[keep] - center) @ R.T, colors[keep]
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def camera_centers(poses):
    return np.array([(-R.T @ t).ravel() for R, t in poses])


def scene_axes(poses):
    """Average up and (horizontal) viewing direction of the real cameras. Image y points down, so up = -row 1 of R."""
    up = -np.mean([R[1] for R, _ in poses], axis=0)
    up /= np.linalg.norm(up)
    forward = np.mean([R[2] for R, _ in poses], axis=0)
    forward -= (forward @ up) * up
    return up, forward / np.linalg.norm(forward)


def orbit_camera(points, poses, angle_deg, width, height, elevation_deg=30):
    """Virtual camera on a circle around the scene, looking at its centre, with 'up' from scene_axes."""
    up, forward = scene_axes(poses)
    side = np.cross(forward, up)

    target = np.median(points, axis=0)
    radius = np.percentile(np.linalg.norm(points - target, axis=1), 85)
    radius = max(radius, np.linalg.norm(camera_centers(poses) - target, axis=1).max())
    a, e = np.radians(angle_deg - 30), np.radians(elevation_deg)
    eye = target + 2.0 * radius * (np.cos(e) * (-np.cos(a) * forward + np.sin(a) * side) + np.sin(e) * up)

    z = (target - eye) / np.linalg.norm(target - eye)   # viewing direction
    y = -(up - (up @ z) * z)                             # image 'down'
    y /= np.linalg.norm(y)
    R = np.stack([np.cross(y, z), y, z])
    f = 0.9 * width
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
    return R, -R @ eye, K


def render(points, colors, poses, R, t, K, width, height):
    """Project all points with x = K (R X + t) and draw them far-to-near (painter's algorithm)."""
    img = np.full((height, width, 3), 18, dtype=np.uint8)
    Xc = points @ R.T + t
    uv = Xc @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    visible = (Xc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    for i in np.flatnonzero(visible)[np.argsort(-Xc[visible, 2])]:
        cv2.circle(img, (int(uv[i, 0]), int(uv[i, 1])), 2, colors[i].tolist(), -1, cv2.LINE_AA)

    c = camera_centers(poses) @ R.T + t
    if np.all(c[:, 2] > 0):  # recorded camera path in orange
        c = c @ K.T
        path = np.round(c[:, :2] / c[:, 2:3]).astype(np.int32)
        cv2.polylines(img, [path], False, (255, 140, 60), 2, cv2.LINE_AA)
        for p in path:
            cv2.circle(img, tuple(int(v) for v in p), 4, (255, 107, 74), -1, cv2.LINE_AA)
    return img


def save_video(path, points, colors, poses, n_frames=180, size=(1280, 720)):
    frames = []
    for i in range(n_frames):
        R, t, K = orbit_camera(points, poses, 360 * i / n_frames, *size)
        frames.append(render(points, colors, poses, R, t, K, *size))
    imageio.mimsave(path, frames, fps=30, macro_block_size=8)

