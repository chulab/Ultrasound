"""Tests for `elastic_image.py`"""

import tensorflow as tf

import elastic_image


class elasticImageTests(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testElasticImage(self):
    image = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
    control_points = tf.constant([[0, 0], [1,1]])
    var = tf.constant([1,2,3])

    ei = elastic_image.ElasticImage(image, control_points)
    ei.add_to_variable_dict(var, "test_type")

    image_test = ei.image
    control_points_test = ei.control_points
    var_test_list = ei.get_list_from_variable_dict("test_type")

    with self.test_session() as sess:
      self.assertAllClose(
        image.eval(),
        image_test.eval()
      )
      self.assertAllClose(
        control_points.eval(),
        control_points_test.eval()
      )
      self.assertAllClose(
        var_test_list[0].eval(),
        var.eval()
      )


if __name__ == '__main__':
  tf.test.main()
