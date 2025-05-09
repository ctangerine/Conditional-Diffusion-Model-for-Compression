import cv2
import os
from typing import List
from tqdm import tqdm

class LosslessVideoGenerator:
    def __init__(self, frame_folder: str, output_path: str, frame_rate: int = 15) -> None:
        """
        Initializes the video generator.

        :param frame_folder: Absolute path to the folder containing video frames.
        :param output_path: Absolute path to save the generated video.
        :param frame_rate: Frame rate of the output video.
        """
        self.frame_folder: str = frame_folder
        self.output_path: str = output_path
        self.frame_rate: int = frame_rate

    def generate_video(self, time_annotation: bool = True, fps: int = 30) -> None:
        """
        Generates a lossless video from the frames in the specified folder.
        Optionally adds time and frame number annotations.
        Ensures the output is saved in .avi format using the MJPEG codec.
        """
        self.frame_rate = fps
        # Get a sorted list of frame file names
        frames: List[str] = sorted([f for f in os.listdir(self.frame_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if not frames:
            raise ValueError("No image frames found in the specified folder.")

        # Read the first frame to get the video dimensions
        first_frame_path: str = os.path.join(self.frame_folder, frames[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            raise ValueError(f"Unable to read the first frame: {first_frame_path}")

        height, width, _ = first_frame.shape

        # Ensure the output file has .avi extension
        if not self.output_path.lower().endswith('.avi'):
            raise ValueError("Output file must have a .avi extension for MJPEG codec.")

        # Define the video writer with MJPEG codec for lossless compression
        fourcc: int = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(self.output_path, fourcc, self.frame_rate, (width, height))

        # Write each frame to the video with a progress bar
        for frame_index, frame_name in enumerate(tqdm(frames, desc="Generating video", unit="frame")):
            frame_path: str = os.path.join(self.frame_folder, frame_name)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Skipping unreadable frame: {frame_path}")
                continue

            # Add time and frame number annotation if enabled
            if time_annotation:
                timestamp = f"Time: {frame_index / self.frame_rate:.2f}s Frame: {frame_index}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text_size, _ = cv2.getTextSize(timestamp, font, font_scale, font_thickness)
                text_x, text_y = 10, 20  # Top-left corner
                cv2.putText(frame, timestamp, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            video_writer.write(frame)

        # Release the video write1r
        video_writer.release()
        print(f"Lossless video saved to: {self.output_path}")

if __name__ == "__main__":
    # Example usage
    frame_folder = "D:\\videos\\VisDrone2019-VID-val\\sequences\\uav0000182_00000_v"
    output_path = "D:\\videos\\video2.avi"
    video_generator = LosslessVideoGenerator(frame_folder, output_path)
    video_generator.generate_video(True, 15)