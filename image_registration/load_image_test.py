"""Tests for `loss_utils.py`"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from image_registration import load_image

class testLoadImage(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testLoadImageBadArgs(self):
    with self.assertRaises(ValueError):
      load_image.load_image(
        tf.ones([5, 5]),
        tf.ones([3,4,5]),
        3., [2, 5.], 1.,
      )

  def testLoadImage(self):
    elastic_image = load_image.load_image(
        tf.ones([5, 5]),
        tf.ones([3, 2]),
        3., [2, 5.], 1.,
      )

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      self.assertAllClose(elastic_image.image.eval(),
                          np.ones([5, 5]))
      self.assertAllClose(elastic_image.control_points.eval(),
                          np.ones([3,2]))
      self.assertAllClose(elastic_image.center_point.eval(),
                          [2.5, 2.5])
      self.assertAllClose(elastic_image.rotation.eval(), 3.)
      self.assertAllClose(elastic_image.translation.eval(), [2, 5.])
      self.assertEqual(elastic_image.non_rigid.shape, (3, 2))

if __name__ == '__main__':
      tf.test.main()
