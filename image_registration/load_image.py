"""Defines ImageField class."""

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

  if len(control_points.shape) != 2 or control_points.shape[1] != 2:
    raise ValueError("""`control_points` must have shape [N, 2] got \
      {}""".format(control_points.shape))

  eimage = elastic_image.ElasticImage(image, control_points, )

  rotation = warp_parameters.make_rotation_warp_variable(initial_rotation, )
  eimage.add_to_variable_dict(rotation, "rotation")

  translation = warp_parameters.make_translation_warp_variable(
    initial_translation, )
  eimage.add_to_variable_dict(translation, "translation")

  non_rigid = warp_parameters.make_elastic_warp_variable(
    control_points.get_shape().as_list()[0], initial_non_rigid_values_or_scale)
  eimage.add_to_variable_dict(non_rigid, "non_rigid")

  return eimage