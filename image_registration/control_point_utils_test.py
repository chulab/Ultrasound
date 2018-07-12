"""Tests for `control_point_utils.py`"""

import tensorflow as tf

import control_point_utils


class imageProcessingUtilsTests(tf.test.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_project_rotation_on_control_points(self):

    control_points = tf.constant([[1., 0],
                                  [0., 1.]])

    center_point = tf.constant([1., 0])

    rotation = tf.constant(30.)

    rotated_points = \
      control_point_utils.project_rotation_on_control_points(
        control_points, center_point, rotation)

    with self.test_session() as sess:
      rotated_point_eval = rotated_points.eval()

    self.assertAllClose(
      rotated_point_eval,
      [[0, 0.],
       [-0.366, -0.634]],
      atol = .001
    )


if __name__ == '__main__':
  tf.test.main()
