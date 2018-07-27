from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from collections import defaultdict

tfe = tf.contrib.eager

class ElasticImage():
  '''An ElasticImage contains an image and associated registration parameters.

  Args:
    image: tf.Tensor of shape [H, W]
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

    self.image = tf.convert_to_tensor(image)
    self.control_points = tf.convert_to_tensor(control_points)
    self.center_point = tf.cast(tf.shape(self.image) / 2., tf.float32)
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
  def control_points(self, control_points):
    self._control_points = control_points

  @property
  def center_point(self):
    return self._center_point

  @center_point.setter
  def center_point(self, center_point):
    self._center_point = center_point

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, name):
    self._name = name

  def add_to_variable_dict(self,
                           variable,
                           type=None):
    '''Adds a new variable to the variable dictionary.'''
    self.variable_dict['all_variables'].append(variable)

    if type is not None:
      assert isinstance(type, str)
      self.variable_dict[type].append(variable)

  def get_list_from_variable_dict(self,
                                  type='all_variables'):
    assert type in self.variable_dict
    return self.variable_dict[type]
