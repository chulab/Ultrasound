"""Tests for `elastic_image.py`"""

import tensorflow as tf

from image_registration import elastic_image


class elasticImageTests(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()
    self.image = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    self.control_points = tf.constant([[0, 0], [1, 1]])
    self.ei = elastic_image.ElasticImage(self.image, self.control_points)

  def testElasticImageBadControlPoints(self):
    image = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
    control_points = tf.constant([[0], [1]])
    with self.assertRaises(ValueError):
       elastic_image.ElasticImage(image, control_points)

  def testElasticImage(self):
    with self.test_session() as sess:
      self.assertAllClose(
        self.image.eval(),
        self.ei.image.eval()
      )
      self.assertAllClose(
        self.control_points.eval(),
        self.ei.control_points.eval()
      )

  def testElasticImageBadRotation(self):
    with self.assertRaises(ValueError):
      self.ei.rotation = tf.constant([1.])

  def testElasticImageRotation(self):
    self.ei.rotation = tf.constant(1.)
    with self.test_session() as sess:
      self.assertAllClose(
        self.ei.rotation.eval(),
        1.
      )

  def testElasticImageBadTranslation(self):
    with self.assertRaises(ValueError):
      self.ei.translation = tf.constant([1., 2, 3])

  def testElasticImageTranslation(self):
    self.ei.translation = tf.constant([1., 2.])
    with self.test_session() as sess:
      self.assertAllClose(
        self.ei.translation.eval(),
        [1., 2]
      )


if __name__ == '__main__':
  tf.test.main()
