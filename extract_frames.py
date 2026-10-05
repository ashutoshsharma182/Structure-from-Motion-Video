"""
Step 1: save frames from a video at a low rate, so consecutive frames differ enough in viewpoint.

    python extract_frames.py --video data/videos/my_video.MOV --out data/my_video --fps 2

Too high an fps means tiny camera motion between frames (little parallax, noisy depth);
too low means too little overlap (matching fails). Moving slowly at walking pace, 2-3 fps is a good start.
"""
import argparse
import os

import imageio.v3 as iio

parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--fps', type=float, default=2.0, help="frames to keep per second of video")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
video_fps = iio.immeta(args.video)['fps']
step = max(1, round(video_fps / args.fps))
count = 0
for i, frame in enumerate(iio.imiter(args.video)):  # phone rotation metadata is applied automatically
    if i % step == 0:
        iio.imwrite(os.path.join(args.out, f"{i:05d}.png"), frame)
        count += 1
print(f"Saved {count} frames (every {step}th frame of a {video_fps:.0f} fps video) to {args.out}/")
