"Tests for steering_utils.py"

import steering_utils

import numpy as np
import unittest


class testSteeringUtils(unittest.TestCase):

    def testCalculatAngleNoFinalX(self):
        real_angle = .245

        point = [5, 7]
        rotation_center = [-3, 5]

        calculated_angle = steering_utils.calculate_angle(point, rotation_center)



    def testCalculateAngle(self):
        real_angle = .617

        point = [5, 7]
        rotation_center = [-3, 5]
        final_x = 2

        calculated_angle = steering_utils.calculate_angle(point, rotation_center, final_x)

        self.assertAlmostEqual(calculated_angle, real_angle, 3)


class testCalculateRotationPoint(unittest.TestCase):

    def testCalculateRotationPointTrivial(self):
        point_1 = np.array([1, 0])
        point_2 = np.array([0, 1])
        rotation = np.pi/2

        real_rp = [0, 0]

        calculated_rp = steering_utils.calculate_rotation_point(point_1, point_2, rotation)

        np.testing.assert_almost_equal(calculated_rp, real_rp)

    def testCalculateRotationPoint(self):
        point_1 = np.array([9., 2.])
        point_2 = np.array([7.6, 11.3])
        rotation = .78

        real_rp = [-3, 5]

        calculated_rp = steering_utils.calculate_rotation_point(point_1, point_2,
                                                              rotation)

        np.testing.assert_almost_equal(calculated_rp, real_rp, 1)


if __name__ == "__main__":
    unittest.main()