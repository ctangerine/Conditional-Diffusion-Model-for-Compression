import numpy as np

"""
This file contains serveral methods for calculating the difference between frames.
"""

def mse(a: np.array, b: np.array) -> float:
    """
    Calculate the mean square error between two frames.
    :param a: The first frame
    :param b: The second frame
    :return: The mean square error between the two frames
    """
    return np.mean((a.astype(np.int16) - b.astype(np.int16)) ** 2)

def 