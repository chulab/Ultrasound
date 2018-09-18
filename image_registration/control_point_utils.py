"""Utilities for working with image control points.

All utilities use array indexing (also known as i-j indexing as opposed to
x-y indexing). For example, the positions [1,2] and [4, 3] will correspond to:

  . . . . .
  . . x . .
  . . . . .
  . . . . .
  . . . x .
"""

import tensorflow as tf

def project_rotation_on_control_points(
    control_points: tf.Tensor,
    center_point: tf.Tensor,
    rotation: tf.Tensor,
) -> tf.Tensor:
  """Calculates displacement of control_points based on rotation.

  Calculates the displacement of each control point as they are rotated about
  `center_point` by rotation `rotation` in a counter-clockwise direction.

  Note that this is not the location of the `control_points` after rotation.
  Rather, it is the vector between `control_points` and their rotated image.
  Explicitly,

     = control_point_image[i] - control_point[i]

  Args:
    control_points: `tf.Tensor` of shape `[batch_size, num_points, 2]`.
    center_point: `tf.Tensor` compatible with `control_points`.
    rotation: Scalar `tf.Tensor`.  Represents rotation in degrees clockwise.

  Returns:
    tf.Tensor of same shape as `control_points`.
  """
  # Center coordinate grid at `center_point`.
  centered_control_points = control_points - center_point

  rotation_radians = rotation * 2 * 3.1415 / 360

  rotation_matrix = tf.reshape(tf.stack([
      tf.cos(rotation_radians),
      - tf.sin(rotation_radians),
      tf.sin(rotation_radians),
      tf.cos(rotation_radians),
  ]),
    shape=[-1, 2, 2])

  centered_rotation_points = tf.einsum(
    'bij,bnj->bni', rotation_matrix, centered_control_points)

  return centered_rotation_points - centered_control_points