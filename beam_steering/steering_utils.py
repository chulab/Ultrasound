"""Functions to guide beam steering operations.

    The coordinate system used is as shown.

    ---->  X-hat
    |
    |
    v
    Z-hat

    Angles are measured in radians counter-clockwise about Z-hat x X-hat where
    0 is parallel to z-hat.

    """

import numpy as np
import math
from typing import List

def calculate_angle(
        point: List[int],
        rotation_center: List[int],
        final_x: int = None,
):
    """Calculates steering angle to move a pixel to a new horizontal location.

    Args:
        point: List of shape [Z, X] corresponding to pixel to be moved.
        rotation_center: List of shape [Z, X] corresponding to pixel to be
            moved.
        final_x: integer corresponding to desired final horizontal (Y)
            position of the pixel.
    Returns:
        Angle to rotate beam.
    """
    point = np.array(point, dtype=np.float32)
    rotation_center = np.array(rotation_center, dtype=np.float32)

    distance = np.sqrt(np.sum((point - rotation_center) ** 2))

    zeroing_angle = math.asin((point[1] - rotation_center[1]) / distance)

    if final_x:
        offset_angle = math.asin((rotation_center[1]-final_x) / distance)
    else:
        offset_angle=0

    return zeroing_angle + offset_angle


def calculate_rotation_point(point_1: np.ndarray,
                           point_2: np.ndarray,
                           rotation: float,
                           ):
    """Calculates location of rotation point.

    Let point 1 and point 2 be the same delta function observed when
    the coordinate system is rotated by a known amount about a rotation
    point (rp).

    This method calculates the rotation point (rp) by finding:

    [1 - R(rotation)](x - R(rotation)x)

    where R(\theta) is the 2D rotation matrix corresponding to a rotation
    by angle theta.

    Args:
        point_1: Initial location of delta_fn point. np.ndarray of form [Z, X].
        point_2: Second location of delta_fn point. np.ndarray of form [Z, X].
        rotation: Amount coordinates are rotated.

    Returns:
        Coordinates of rotation point.

    Raises:
        ValueError: If rotation is invalid.
    """
    rotation_matrix = np.array([[math.cos(rotation), -math.sin(rotation)],
                                [math.sin(rotation), math.cos(rotation)]])

    try:
        inverted_matrix = np.linalg.inv(np.identity(2) - rotation_matrix)
    except np.linalg.LinAlgError:
        raise  ValueError("1 - rotation matrix is singular.")

    return np.dot(inverted_matrix,
                  (point_2 - np.dot(rotation_matrix, point_1)))