"""Tests for `image_processing_utils.py`"""

import tensorflow as tf

import image_processing_utils


class imageProcessingUtilsTests(tf.test.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_calculate_pad_size(self):
    array_shape = [5, 7]
    new_shape = [21, 15]
    offsets = [2, -2]

    padding, corner_position, center_position = \
      image_processing_utils.calculate_pad_size(array_shape, new_shape,
                                                offsets)

    with self.test_session() as sess:
      padding_eval, corner_position_eval, center_position_eval = \
        sess.run([padding, corner_position, center_position])

    self.assertAllEqual(padding_eval, [[10, 6], [2, 6]])
    self.assertAllEqual(corner_position_eval, [[[10, 2], [10, 9]],
                                            [[15, 2], [15, 9]], ])
    self.assertAllClose(center_position_eval, [12.5, 5.5])


if __name__ == '__main__':
  tf.test.main()
