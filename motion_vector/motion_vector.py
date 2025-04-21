import numpy as np

"""
This file is for the motion vector module, supporting the detection of redundant frames
"""

def frames_differences(start_frame: np.array, end_frame: np.array, method: str = "mse") -> np.array:
    """
    Calculate the difference between two frames.
    :param start_frame: The first frame
    :param end_frame: The second frame
    :return: The difference between the two frames
    """
    return np.abs(start_frame.astype(np.int16) - end_frame.astype(np.int16))

"""
TODO: Find the suitable video format
"""