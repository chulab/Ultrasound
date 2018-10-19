"""Image loading utils."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import Union, List

import tensorflow as tf

from image_registration import elastic_image
from image_registration import warp_parameters


def load_image(
    image: tf.Tensor,
    control_points: tf.Tensor,
    initial_rotation: float,
    initial_translation: List[float],
    initial_non_rigid_values_or_scale: Union[List[float], float]
):
  """Creates ElasticImage object with variables for registration.

  Convenience function that wraps instantiation of `ElasticImage` as well
  as variable creation using `warp_parameters`.

  Args:
    image: `tf.Tensor` of shape `[height, width]`.
    control_points: `tf.Tensor` of shape `[num_control_points, 2]`. Describes
      locations of control points in image used to parametrize warp.
    initial_rotation: Initial rotation of image in degrees.
    initial_translation: Initial translation of image.
    initial_non_rigid_values_or_scale: Either a List of initial displacements
      control_point grid or a float which parametrizes the scale of random
      initialization. See `warp_parameters.make_elastic_warp_variable`.

  Returns:
    ElasticImage object containing image and variables for rotation,
      translation, and non-rigid warp.

  Raises:
    ValueError: if `control_points` has an invalid shape.

  """

  if not control_points.shape.is_compatible_with([None, 2]):
    raise ValueError("`control_points` must be compatible with [None, 2]"
      "got {}".format(control_points.shape))

  eimage = elastic_image.ElasticImage(image, control_points)

  control_points_shape = eimage.control_points.shape

  rotation = warp_parameters.make_rotation_warp_variable(initial_rotation)
  eimage.rotation = rotation

  translation = warp_parameters.make_translation_warp_variable(
    initial_translation
  )
  eimage.translation = translation

  non_rigid = warp_parameters.make_elastic_warp_variable(
    control_points_shape, initial_non_rigid_values_or_scale)
  eimage.non_rigid = non_rigid

  return eimage