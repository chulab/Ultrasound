"""Tests for `control_point_utils.py`"""

import tensorflow as tf
import numpy as np

from image_registration import control_point_utils
from image_registration import elastic_image

class imageProcessingUtilsTests(tf.test.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def testWarpTensor(self):
    eim = elastic_image.ElasticImage(tf.ones([3,4]), tf.ones([5,2]))
    eim.rotation = tf.constant(0.)
    eim.translation = tf.constant([3,5])
    eim.non_rigid = tf.ones([5,2]) * 4
    tensors = control_point_utils.warp_tensor(eim, True, True, True)
    with self.test_session():
      self.assertAllClose(tensors[0].eval(), np.zeros([1, 5, 2]))
      self.assertAllClose(tensors[1].eval(), np.array([[[3, 5]]]))
      self.assertAllClose(tensors[2].eval(), np.ones([1, 5,2]) * 4)

  def testWarpVariable(self):
    eim = elastic_image.ElasticImage(tf.ones([3,4]), tf.ones([5,2]))
    eim.rotation = tf.constant(7.)
    eim.translation = tf.constant([3,5])
    eim.non_rigid = tf.zeros([5,2])
    variables = control_point_utils.warp_variables(eim, True, True, True)
    with self.test_session():
      self.assertAllClose(variables[0].eval(), 7)
      self.assertAllClose(variables[1].eval(), [3, 5])
      self.assertAllClose(variables[2].eval(), np.zeros([5, 2]))


  def testPrepareTranslation(self):
    translation = tf.constant([1,2])
    with self.test_session():
      self.assertAllClose(
        control_point_utils.prepare_translation(translation).eval(),
        [[1,2]]
      )

  def testPrepareTranslationBadArgs(self):
    with self.assertRaises(ValueError):
      control_point_utils.prepare_translation(tf.constant([[1, 2]]))

  def testProjectRotationOnControlPoints(self):
    control_points = tf.constant([[1., 0],
                                   [0., 1.]])

    center_point = tf.constant([1., 0])

    rotation = tf.constant(30.)

    rotated_points = (
      control_point_utils.project_rotation_on_control_points(
        control_points, center_point, rotation)
    )

    with self.test_session() as sess:
      rotated_point_eval = rotated_points.eval()

    self.assertAllClose(
      rotated_point_eval,
      [[0, 0.],
        [-0.366, -0.634]],
      atol=.001
    )


if __name__ == '__main__':
  tf.test.main()
