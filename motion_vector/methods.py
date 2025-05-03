import numpy as np
import cv2

"""
This file contains serveral methods for calculating the difference between frames.
"""

import cv2
import numpy as np
import os
from typing import List
import matplotlib.pyplot as plt


import numpy as np
import cv2
from tqdm import tqdm

def get_redundant_frame_indices(frames: np.ndarray,
								 color_space: str = 'YUV',
								 k: float = 1.0,
								 min_spacing: int = 1) -> list:
	"""
	Identifies redundant frames in a sequence of images.

	Args:
		frames: np.ndarray of shape (N, H, W, C), dtype uint8 or float32
		color_space: One of ['RGB', 'YUV', 'GRAY']
		k: Sensitivity factor for adaptive thresholding.
		min_spacing: Minimum number of frames between two removals.

	Returns:
		List of indices of frames that can be removed.
	"""

	def convert_color(frame, space):
		if space == 'GRAY':
			return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		elif space == 'YUV':
			return cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
		else:
			return frame  # Already RGB

	N = len(frames)
	keep_indices = [0]
	last_kept = convert_color(frames[0], color_space).astype(np.float32)
	diff_scores = []

	for i in range(1, N):
		curr_proc = convert_color(frames[i], color_space).astype(np.float32)
		diff = np.mean(np.abs(last_kept - curr_proc))
		diff_scores.append(diff)

		threshold = np.mean(diff_scores) + k * np.std(diff_scores) if len(diff_scores) > 5 else 5.0

		if diff > threshold or i - keep_indices[-1] < min_spacing:
			keep_indices.append(i)
			last_kept = curr_proc

	# Frames not in keep_indices are redundant
	redundant_indices = [i for i in range(N) if i not in keep_indices]
	return redundant_indices, diff_scores


if __name__ == "__main__":
	VIDEO_PATH = "D:\\videos\\video.avi"

	# Read video
	cap = cv2.VideoCapture(VIDEO_PATH)
	if not cap.isOpened():
		raise IOError(f"Cannot open video file: {VIDEO_PATH}")

	frames = []
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	with tqdm(total=total_frames, desc="Reading frames") as pbar:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			frames.append(frame)
			pbar.update(1)
	cap.release()
	frames = np.array(frames)
	print(f"Total frames: {len(frames)}")
	# Get redundant frames
	redundant_indices, diff_score = get_redundant_frame_indices(frames, color_space='YUV', k=1.0, min_spacing=5)
	print(f"Redundant frames: {redundant_indices}")

	# Plot diff scores
	plt.plot(diff_score)
	plt.title("Difference Scores")
	plt.xlabel("Frame Index")
	plt.ylabel("Difference Score")
	plt.show()