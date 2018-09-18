"""tests for warp.py"""

import numpy as np
import tensorflow as tf
import warp


class TestWarp(tf.test.TestCase):
  def setUp(self):
    tf.reset_default_graph()
    self.image_np = np.array([[5, 5, 5, 5, 5, 5, 5, 5],
                             [5, 5, 5, 5, 5, 5, 5, 5],
                             [5, 5, 0, 1, 2, 3, 5, 5],
                             [5, 5, 4, 5, 6, 7, 5, 5],
                             [5, 5, 8, 9, 10, 11, 5, 5],
                             [5, 5, 5, 5, 5, 5, 5, 5],
                             [5, 5, 5, 5, 5, 5, 5, 5], ])
    self.image_np = self.image_np[np.newaxis, :, :, np.newaxis]

    self.image = tf.constant(self.image_np)



  def testDenseWarp(self):
    """Tests dense_warp"""

    correct_warp = np.array([[5, 5, 5, 5, 5, 5, 5, 5],
                             [5, 5, 5, 5, 5, 5, 5, 5],
                             [5, 5, 5, 5, 5, 5, 5, 5],
                             [5, 5, 0, 1, 2, 3, 5, 5],
                             [5, 5, 4, 5, 6, 7, 5, 5],
                             [5, 5, 8, 9, 10, 11, 5, 5],
                             [5, 5, 5, 5, 5, 5, 5, 5], ])

    warp_matrix = tf.constant([[[[1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.]],

                                [[1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.]],

                                [[1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.]],

                                [[1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.]],

                                [[1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.]],

                                [[1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.]],

                                [[1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.],
                                 [1., 0.]], ]])

    warped_image = warp.dense_warp(self.image, [warp_matrix])
    with tf.Session() as sess:
      warped_image_eval = sess.run(warped_image)
    self.assertAllClose(warped_image_eval[0, :, :, 0], correct_warp)

  def testDenseWarpNoWarp(self):
    """Tests dense_warp"""

    warp_matrix = tf.constant([[[[0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.]],

                                [[0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.]],

                                [[0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.]],

                                [[0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.]],

                                [[0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.]],

                                [[0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.]],

                                [[0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.],
                                 [0., 0.]]]])

    warped_image = warp.dense_warp(self.image, [warp_matrix])
    with tf.Session() as sess:
      warped_image_eval = sess.run(warped_image)
    self.assertAllClose(warped_image_eval, self.image_np)

  def testWarpQueryNoScaleNoWarp(self):
    control_points = tf.constant([[[0,0], [0, 7], [6, 0], [6,7]]])
    warp_values = [tf.constant([[[0,0]] * 4 ])]

    query_points = tf.constant([[[0,0], [2,2], [4,3]]])

    query_val = warp.warp_query(self.image, query_points, control_points,
                                warp_values)

    truth_vals = [[[5], [0], [9]]]

    with self.test_session() as sess:
      self.assertAllClose(query_val.eval(), truth_vals)

  def testWarpQueryNoScale(self):
    control_points = tf.constant([[[0, 0], [0, 7], [6, 0], [6, 7]]])
    warp_values = [tf.constant([[[1, 1]] * 4])]

    query_points = tf.constant([[[0, 0], [2, 2], [3,3], [4, 3]]])

    query_val = warp.warp_query(self.image, query_points, control_points,
                                warp_values)

    truth_vals = [[[5], [5], [0], [4]]]

    with self.test_session() as sess:
      self.assertAllClose(query_val.eval(), truth_vals)

  def testWarpQuery(self):
    control_points = tf.constant([[[0, 0], [0, 7], [6, 0], [6, 7]]])
    warp_values = [tf.constant([[[2, 2]] * 4])]

    query_points = tf.constant([[[0, 0], [2, 2], [4,4], [4, 6], [3, 3]]])

    query_val = warp.warp_query(self.image, query_points, control_points,
                                warp_values, 2)

    truth_vals = [[[5], [5], [2.5], [4.5], [4.375]]]

    with self.test_session() as sess:
      self.assertAllClose(query_val.eval(), truth_vals)

  def testTotalWarpPoints(self):
    total_points = warp.total_warp_points(
      [tf.constant([[[1, 2], [3, 4], [5, 6]]]),
       tf.constant([[[1, 1], [2, 2], [3, 3]]])],
      tf.TensorShape([1, 3, 2])
    )
    with self.test_session() as sess:
      self.assertAllEqual(total_points.eval(),
                          np.array([[[2, 3], [5, 6], [8, 9]]]))

  def testTotalWarpPointsBadArgs(self):
    with self.assertRaisesRegex(ValueError, "[1, 2, 2]"):
      warp.total_warp_points(
        [tf.constant([[[1,2], [3,4], [5,6]]]),
         tf.constant([[[1,1], [2,2]]])],
        tf.TensorShape([1, 3, 2])
      )

  def testRescalePoints(self):
    points = tf.constant([1., 2, 3])
    scale = 2.
    rescaled_points = warp.rescale_points(points, scale)
    with self.test_session() as sess:
      rescaled_points_eval = sess.run(rescaled_points)
    self.assertAllClose(rescaled_points_eval, [.5, 1., 1.5])

  def testRescaleImageUsingPooling(self):
    image = tf.constant(np.ones([1, 10, 10, 1]))
    resized_image = warp.rescale_image_using_pooling(image, 2)
    self.assertEqual(resized_image.shape.as_list(), [1, 5, 5, 1])
    with self.test_session() as sess:
        self.assertAllEqual(sess.run(resized_image), np.ones([1, 5, 5, 1]))

if __name__ == '__main__':
  tf.test.main()
