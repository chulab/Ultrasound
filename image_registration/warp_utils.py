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
from image_registration import elastic_image


def warp_tensor(
  elastic_image: elastic_image.ElasticImage,
  rotation: bool,
  translation: bool,
  non_rigid: bool
):
  """Prepares variables from `ElasticImage` for warp operations.

  Same `warp_variables` but prepares each variable for warp operations by using
  the apropriate preparation function from `warp_utils`.

  Note that this function returns objects from both `tf.Tensor` and
  `tf.Variable` nodes depending on which variables are requested.

  Args:
    elastic_image: `ElasticImage` object.
    rotation: bool. If `True` then the prepared rotation variable is added to
      the list of returned tensors.
    translation: Same as `rotation` but for translation variable.
    non_rigid: Same as `rotation` but for non_rigid variable.

  Returns:
    variables: List of `tf.Tensor` corresponding requested variables.
  """
  tensors = []

  if rotation:
    tensors += [rotate_points(
      elastic_image.control_points, elastic_image.center_point,
      elastic_image.rotation)]
  if translation:
    tensors += [prepare_translation(elastic_image.translation)]
  if non_rigid:
    tensors += [elastic_image.non_rigid]

  # Update shape of tensors so they have correct dimension for warp functions.
  # [batch_size, point_count, 2]
  return [warp_tensor[tf.newaxis, :, :] for warp_tensor in tensors]


def warp_variables(
  elastic_image: elastic_image.ElasticImage,
  rotation: bool,
  translation: bool,
  non_rigid: bool
):
  """Retrieves variables from ElasticImage.

  This utility function populates a list of warp variables from an
  `ElasticImage`.

  Args:
    elastic_image: `ElasticImage` object.
    rotation: bool. If `True` then the rotation variable is added to the list
    of returned variables.
    translation: Same as `rotation` but for translation variable.
    non_rigid: Same as `rotation` but for non_rigid variable.

  Returns:
    variables: List of `tf.Variable` corresponding to selected variable types.
  """
  variables = []

  if rotation:
    variables += [elastic_image.rotation]
  if translation:
    variables += [elastic_image.translation]
  if non_rigid:
    variables += [elastic_image.non_rigid]
  return variables


def prepare_translation(
    translation: tf.Tensor,
):
  """Makes `translation` compatible with `control_points`."""
  translation.shape.assert_is_compatible_with([2])
  return translation[tf.newaxis, :]


def rotate_points(
    points: tf.Tensor,
    center_point: tf.Tensor,
    rotation: tf.Tensor,
) -> tf.Tensor:
  """Calculates displacement of points based on rotation.

  Calculates the displacement of each point as they are rotated about
  `center_point` by rotation `rotation` in a counter-clockwise direction.

  Note that this is not the location of the `points` after rotation.
  Rather, it is the vector between `points` and their rotated image.
  Explicitly,

     = control_point_image[i] - control_point[i]

  Args:
    points: `tf.Tensor` of shape `[batch_size, num_points, 2]`.
    center_point: `tf.Tensor` compatible with `points`.
    rotation: Scalar `tf.Tensor`.  Represents rotation in degrees clockwise.

  Returns:
    tf.Tensor of same shape as `points`.

  Raises:
    ValueError: If `rotation` is not scalar.
  """
  rotation.shape.assert_is_compatible_with([])
  points.shape.assert_is_compatible_with([None, 2])

  # Center coordinate grid at `center_point`.
  centered_points = points - center_point

  rotation_radians = rotation * 2 * 3.1415 / 360

  rotation_matrix = tf.reshape(tf.stack([
    tf.cos(rotation_radians),
    - tf.sin(rotation_radians),
    tf.sin(rotation_radians),
    tf.cos(rotation_radians),
  ]),
    shape=[2, 2])

  rotation_matrix = tf.cast(rotation_matrix, centered_points.dtype)

  centered_rotation_points = tf.einsum(
    'ij,nj->ni', rotation_matrix, centered_points)

  return centered_rotation_points - centered_points
