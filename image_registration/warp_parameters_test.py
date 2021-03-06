"""Tests for `warp_parameters.py`"""

import tensorflow as tf

from image_registration import warp_parameters

class warpParametersTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_make_elastic_warp_variable(self):

    control_point_shape = tf.TensorShape([1, 20, 2])

    elastic_warp = warp_parameters.make_elastic_warp_variable(
      control_point_shape, 3.)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      elastic_warp_eval = elastic_warp.eval()

    self.assertAllClose(elastic_warp_eval, [[[0., 0.]] * 20], atol=3. )

  def test_make_elastic_warp_variable_initialized(self):

    control_point_shape = tf.TensorShape([1, 20, 2])

    elastic_warp = warp_parameters.make_elastic_warp_variable(
      control_point_shape, [[[1., 4.]] * 10])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      elastic_warp_eval = elastic_warp.eval()

    self.assertAllClose(elastic_warp_eval, [[[1., 4.]] * 10])

  def test_make_rotation_warp_variable(self):

    rotation = warp_parameters.make_rotation_warp_variable(10.)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      rotation_eval = rotation.eval()

    self.assertAllClose(rotation_eval, 10.)

  def testMakeRotationWarpVariableBadArgs(self):
    with self.assertRaises(ValueError):
      warp_parameters.make_rotation_warp_variable([10, 2])

  def test_make_translation_warp_variable(self):

    translation = warp_parameters.make_translation_warp_variable([1.,2.])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      translation_eval = translation.eval()

    self.assertAllClose(translation_eval, [1., 2.])

  def testMakeTranslationWarpVariableBadArgs(self):
    with self.assertRaises(ValueError):
      warp_parameters.make_translation_warp_variable([10, 2, 3])

  def testMakeTranslationWarpVariableBadArgsListofList(self):
    with self.assertRaises(ValueError):
      warp_parameters.make_translation_warp_variable([[1], [2]])

if __name__ == '__main__':
  tf.test.main()
