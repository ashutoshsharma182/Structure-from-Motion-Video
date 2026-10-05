"""
Step 2: reconstruct camera poses and a coloured 3D point cloud from ordered frames.

    python reconstruct.py --frames data/my_video --out outputs/my_video

Intrinsics: K.txt inside the frames folder if present, else a rough guess of 1.2 x the longer image side.
Bundle adjustment then refines the focal length together with the cameras and points.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from output import save_ply, save_video
from sfm import reconstruct

parser = argparse.ArgumentParser()
parser.add_argument('--frames', required=True, help="folder of images in capture order")
parser.add_argument('--out', required=True)
parser.add_argument('--max-size', type=int, default=1600, help="downscale images to this longest side")
parser.add_argument('--eps', type=float, default=1.5, help="RANSAC inlier threshold in pixels")
args = parser.parse_args()

names = sorted(f for f in os.listdir(args.frames) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
images = [cv2.imread(os.path.join(args.frames, f)) for f in names]
h, w = images[0].shape[:2]

k_file = os.path.join(args.frames, 'K.txt')
if os.path.exists(k_file):
    K = np.loadtxt(k_file)
else:
    f = 1.2 * max(w, h)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

scale = min(1.0, args.max_size / max(w, h))  # smaller images: faster, K scales with them
images = [cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for img in images]
K[:2] *= scale
print(f"{len(images)} frames, initial focal length {K[0, 0] / scale:.0f} px")

poses, points, colors, f = reconstruct(images, K, eps=args.eps)
print(f"Focal length after bundle adjustment: {f / scale:.0f} px")

os.makedirs(args.out, exist_ok=True)
save_ply(os.path.join(args.out, 'points.ply'), points, colors, poses)
save_video(os.path.join(args.out, 'pointcloud_360.mp4'), points, colors, poses)
print(f"{len(poses)}/{len(images)} cameras, {len(points)} points -> {args.out}/ (points.ply, pointcloud_360.mp4)")
