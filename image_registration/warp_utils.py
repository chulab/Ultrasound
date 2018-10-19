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

_RADIANS_PER_DEGREE = 2 * 3.1415 / 360.


def warp_tensor_dense(
    elastic_image: elastic_image.ElasticImage,
    rotation: bool,
    translation: bool,
    non_rigid: bool
):
  """Prepares Variables from `ElasticImage` for dense warp operations.

  This function should be used when `warp_tenser` are being passed
  directly to `tf.dense_warp`.

  The difference between this and `warp_tensor` arises from the fact that
  `tf.sparse_warp` expects a warp field that points from the `control_points`
  to their image in the warped coordinates.  On the other hand `tf.dense_warp`
  expects a warp field that ends in the coordinates of the warped image.

  Args:
    elastic_image: See `warp_tensor`.
    rotation: bool. See `warp_tensor`.
    translation: See `warp_tensor`.
    non_rigid: See `warp_tensor`.

  Returns:
    warp_field: List of `tf.Tensor` corresponding requested variables.
  """
  tensors = []
  # The main distinction here is that rotation is reversed and the center point
  # is changed so that it is at the translted position. This is necessary
  # when using the dense warp paradigm because we want to perform
  # `translation(rotation(image))` not `rotation(translation(image))`.
  if rotation:
    tensors += [- rotate_points(
      elastic_image.control_points, elastic_image.center_point + elastic_image.translation,
      - elastic_image.rotation)]
  if translation:
    tensors += [prepare_translation(elastic_image.translation)]
  if non_rigid:
    tensors += [elastic_image.non_rigid]

  # Update shape of tensors so they have correct dimension for warp functions.
  # [batch_size, point_count, 2]
  return [warp_tensor[tf.newaxis, :, :] for warp_tensor in tensors]


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

  NOTE that this is not the location of the `points` after rotation.
  Rather, it is the vector between `points` and their rotated image.
  Explicitly,

     = control_point_image[i] - control_point[i]

  Args:
    points: `tf.Tensor` of shape `[num_points, 2]`. Last axis corresponds to
      [i, j] (See note at top of module on coordinates).
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
  center_point = tf.cast(center_point, points.dtype)
  centered_points = points - center_point

  rotation = tf.cast(rotation, tf.float32)
  rotation_radians = rotation * tf.constant(_RADIANS_PER_DEGREE)

  rotation_matrix = tf.reshape(tf.stack([
    tf.cos(rotation_radians),
    - tf.sin(rotation_radians),
    tf.sin(rotation_radians),
    tf.cos(rotation_radians),
  ]),
    shape=[2, 2])

  rotation_matrix = tf.cast(rotation_matrix, points.dtype)

  points_rotated_about_center = tf.einsum(
    'ij,nj->ni', rotation_matrix, centered_points)

  return points_rotated_about_center - centered_points
