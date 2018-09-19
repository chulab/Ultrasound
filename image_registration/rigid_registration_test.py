"""Tests for `rigid_registration.py`"""

import unittest
import numpy as np

from image_registration import rigid_registration

class warpParametersTest(unittest.TestCase):

  def testRelativeRotationAndTranslationBadArgs(self):
    with self.assertRaises(ValueError):
      rigid_registration.relative_rotation_and_translation(
        [np.ones([1,5]), np.ones([3,4])],
        [2., 3],
        4,
      )

if __name__ == '__main__':
  unittest.main()
