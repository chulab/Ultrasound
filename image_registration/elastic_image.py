from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from collections import defaultdict

tfe = tf.contrib.eager

class ElasticImage():
  '''An ElasticImage contains an image and associated registration parameters.

  Args:
    image: `tf.Tensor` of shape [H, W]
    control_points: tf.Tensor of shape [num_points, 2].
    name: str.

  Properties:
    image: tf.Tensor of shape [H, W]
    control_points: tf.Tensor of shape [num_points, 2]
    variable_dict: Dictionary of registration parameters.

  Raises:
    ValueError: If `image` has an invalid shape.
  '''

  def __init__(self,
               image,
               control_points,
               name="elastic_image"
               ):
    self.image = image
    self.control_points = control_points
    self.center_point = tf.cast(tf.shape(image) / tf.constant(2), tf.float32)
    self.name = name

    self.variable_dict = defaultdict(list)

  @property
  def image(self):
    return self._image

  @image.setter
  def image(self, image):
    if len(image.shape) != 2:
      raise ValueError("""Image has incorrect dimension. 
      Should have 2 dimensions, got {}""".format(len(image.get_shape())))
    self._image = image

  @property
  def control_points(self):
    return self._control_points

  @control_points.setter
  def control_points(self, control_points: tf.Tensor):
    control_points.shape.assert_is_compatible_with([None, 2])
    self._control_points = control_points

  @property
  def center_point(self):
    return self._center_point

  @center_point.setter
  def center_point(self, center_point: tf.Tensor):
    self._center_point = center_point

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, name):
    self._name = name

  @property
  def translation(self):
    return self._translation

  @translation.setter
  def translation(self, translation: tf.Tensor):
    translation.shape.assert_is_compatible_with([2])
    self._translation = translation

  @property
  def rotation(self):
    return self._rotation

  @rotation.setter
  def rotation(self, rotation: tf.Tensor):
    rotation.shape.assert_has_rank(0)
    self._rotation = rotation

  @property
  def non_rigid(self):
    return self._non_rigid

  @non_rigid.setter
  def non_rigid(self, non_rigid: tf.Tensor):
    non_rigid.shape.assert_is_compatible_with(self._control_points.shape)
    self._non_rigid = non_rigid
