import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

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

def get_frames_and_diff_scores(frames: np.ndarray,
                                color_space: str = 'YUV',
                                k: float = 1.0,
                                min_spacing: int = 1,
                                output_path: str = 'diff_visualization.mp4',
                                fps: int = 10) -> tuple:
    """
    Generates a video showing each frame alongside its diff score progression.

    Args:
        frames: np.ndarray of shape (N, H, W, C), dtype uint8 or float32
        color_space: One of ['RGB', 'YUV', 'GRAY']
        k: Sensitivity factor for adaptive thresholding.
        min_spacing: Minimum number of frames between two removals.
        output_path: Path to save the output video.
        fps: Frames per second for output video.

    Returns:
        Tuple (redundant_indices, diff_scores)
    """

    def convert_color(frame, space):
        if space == 'GRAY':
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif space == 'YUV':
            return cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
        else:
            return frame  # RGB

    N, H, W, C = frames.shape
    keep_indices = [0]
    last_kept = convert_color(frames[0], color_space).astype(np.float32)
    diff_scores = []

    # Prepare video writer
    fig_width, fig_height = 8, 4  # inches
    dpi = 100
    canvas_width, canvas_height = int(fig_width * dpi), int(fig_height * dpi)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_width, canvas_height))

    for i in tqdm(range(N), desc="Generating visualization video"):
        curr_proc = convert_color(frames[i], color_space).astype(np.float32)
        diff = np.mean(np.abs(last_kept - curr_proc))
        diff_scores.append(diff)

        threshold = np.mean(diff_scores) + k * np.std(diff_scores) if len(diff_scores) > 5 else 5.0

        if diff > threshold or i - keep_indices[-1] < min_spacing:
            keep_indices.append(i)
            last_kept = curr_proc

        # Create visualization frame (in RGB)
        fig = Figure(figsize=(fig_width, fig_height), dpi=dpi)
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(frames[i])  # RGB visualization
        ax1.axis('off')
        ax1.set_title(f"Frame {i}")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(diff_scores, color='blue')
        ax2.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        ax2.set_title("Diff Score")
        ax2.set_xlim(0, N)
        ax2.set_ylim(0, max(diff_scores) + 10)
        ax2.legend()

        canvas.draw()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(canvas_height, canvas_width, 3)
        frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    redundant_indices = [i for i in range(N) if i not in keep_indices]
    return redundant_indices, diff_scores


# if __name__ == "__main__":
# 	VIDEO_PATH = "D:\\videos\\video.avi"

# 	# Read video
# 	cap = cv2.VideoCapture(VIDEO_PATH)
# 	if not cap.isOpened():
# 		raise IOError(f"Cannot open video file: {VIDEO_PATH}")

# 	frames = []
# 	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 	with tqdm(total=total_frames, desc="Reading frames") as pbar:
# 		while True:
# 			ret, frame = cap.read()
# 			if not ret:
# 				break
# 			frames.append(frame)
# 			pbar.update(1)
# 	cap.release()
# 	frames = np.array(frames)
# 	print(f"Total frames: {len(frames)}")
# 	# Get redundant frames
# 	redundant_indices, diff_score = get_redundant_frame_indices(frames, color_space='YUV', k=1.0, min_spacing=5)
# 	print(f"Redundant frames: {redundant_indices}")

# 	# Plot diff scores
# 	plt.plot(diff_score)
# 	plt.title("Difference Scores")
# 	plt.xlabel("Frame Index")
# 	plt.ylabel("Difference Score")
# 	plt.show()

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

	redundants, diff_scores = get_frames_and_diff_scores(frames, color_space='YUV', k=1.0, min_spacing=5, output_path='diff_visualization.mp4', fps=15)