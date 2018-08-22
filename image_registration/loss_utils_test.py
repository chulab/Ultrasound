"""Tests for `loss_utils.py`"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

import loss_utils

class testLossFunctions(tf.test.TestCase):

    def test_masked_mse_no_mask(self):
        a = tf.constant([1, 2, 3])
        b = tf.constant([10, 20, 30])
        c = tf.constant([100, 200, 300])

        loss = loss_utils.masked_mse([a,b,c])

        with self.test_session() as sess:
            loss_eval = sess.run(loss)

        self.assertAllClose(loss_eval, 27972.0)

    def test_masked_mse_mask(self):
        a = tf.constant([1, 2, 3])
        b = tf.constant([10, 20, 30])
        c = tf.constant([100, 200, 300])

        a_mask = tf.constant([1, 0, 1])
        b_mask = tf.constant([1, 1, 1])
        c_mask = tf.constant([1, 1, 0])

        loss = loss_utils.masked_mse([a,b,c], [a_mask, b_mask, c_mask])

        with self.test_session() as sess:
            loss_eval = sess.run(loss)

        self.assertAllClose(loss_eval, 10222.2)

    def test_warp_loss_range_one(self):

        t = tf.constant([])

        loss = loss_utils.warp_loss(t, 1)

        with self.test_session() as sess:
            loss_eval = sess.run(loss)

        print(loss_eval)

        self.assertAllClose(loss_eval, 10222.2)


if __name__ == '__main__':
      tf.test.main()
