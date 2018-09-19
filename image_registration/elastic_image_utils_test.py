"""Tests for `warp_parameters.py`"""

import tensorflow as tf

from image_registration import load_image
from image_registration import elastic_image_utils

class warpParametersTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testReduceRotation(self):
    im_a = load_image.load_image(
      tf.ones([5,5]),
      tf.ones([1, 2]),
      0.,
      [10.,0],
      0.
    )

    im_b = load_image.load_image(
      tf.ones([5, 5]),
      tf.ones([1, 2]),
      90.,
      [0, 10.],
      0.
    )

    update = elastic_image_utils.reduce_rotation([im_a, im_b])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(update)
      self.assertAlmostEqual(
        im_a.rotation.eval(), -45.
      )
      self.assertAlmostEqual(
        im_b.rotation.eval(), 45.
      )
      self.assertAllClose(
        im_a.translation.eval(), [7.07, -7.07], atol=.01
      )
      self.assertAllClose(
        im_b.translation.eval(), [7.07, 7.07], atol=.01
      )

  def testReduceTranslation(self):
    im_a = load_image.load_image(
      tf.ones([5,5]),
      tf.ones([1, 2]),
      0.,
      [0.,0],
      0.
    )

    im_b = load_image.load_image(
      tf.ones([5, 5]),
      tf.ones([1, 2]),
      0.,
      [20., -15.],
      0.
    )

    update = elastic_image_utils.reduce_translation([im_a, im_b])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(update)
      self.assertAllClose(
        im_a.translation.eval(), [-10, 7.5]
      )
      self.assertAllClose(
        im_b.translation.eval(), [10, -7.5]
      )


if __name__ == '__main__':
  tf.test.main()
