#!/usr/bin/env python

"""
@license: MIT
@author: Jan K. Schluesener
https://github.com/jkschluesener/accorr
accorr: tests
"""

# Unit Tests
import unittest
import pydantic
import numpy as np
from numpy.testing import assert_array_almost_equal

from accorr.accorr import accorr
from accorr.models import ComputeInput, ComputeOutput


class Testaccorr(unittest.TestCase):
    def setUp(self):
        self.acc = accorr()
        self.arr_a = np.array(
            [
                [0.49362092, 0.65198427, 0.61944758, 0.33437013, 0.51387025],
                [0.03513959, 0.55706569, 0.00413015, 0.99857145, 0.94614577],
                [0.88532318, 0.16843675, 0.02433818, 0.38443922, 0.51891721],
            ]
        )
        self.arr_b = np.array(
            [
                [0.60303915, 0.48360065, 0.66149025, 0.61038623, 0.23610313],
                [0.21005172, 0.80987631, 0.97749917, 0.67826358, 0.15650298],
                [0.56526303, 0.40989813, 0.55336891, 0.88826624, 0.16354896],
            ]
        )
        self.statistical_threshold = 0.05
        self.feature_labels_a = np.array([1, 2, 3])
        self.feature_labels_b = np.array([4, 5, 6])

    def test_init(self):
        self.assertIsInstance(self.acc, accorr)

    def test_call(self):
        result = self.acc.compute(
            self.arr_a,
            self.arr_b,
            self.statistical_threshold,
            self.feature_labels_a,
            self.feature_labels_b,
        )
        self.assertIsInstance(result, ComputeOutput)

    def test_compute(self):
        result = self.acc.compute(
            self.arr_a,
            self.arr_b,
            self.statistical_threshold,
            self.feature_labels_a,
            self.feature_labels_b,
        )
        self.assertIsInstance(result, ComputeOutput)

    def test__compute(self):
        input_data = ComputeInput(
            arr_a=self.arr_a,
            arr_b=self.arr_b,
            feature_labels_a=self.feature_labels_a,
            feature_labels_b=self.feature_labels_b,
            statistical_threshold=self.statistical_threshold,
        )
        result = self.acc._compute(input_data)
        self.assertIsInstance(result, ComputeOutput)

    def test__reliability(self):
        result = self.acc._reliability(self.arr_a, self.arr_b)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test__rac(self):
        result = self.acc._rac(0.5, 0.6, 0.7)
        self.assertIsInstance(result, float)

    def test__repeated_mean(self):
        result = self.acc._repeated_mean(
            self.arr_a, self.feature_labels_a, self.feature_labels_b
        )
        self.assertIsInstance(result, float)

    def test_ttest(self):
        result = self.acc.ttest(self.arr_a, self.feature_labels_a)
        self.assertIsInstance(result, float)

    def test__ttest(self):
        result = self.acc._ttest(self.arr_a.flatten(), len(self.arr_a.flatten()) - 1)
        self.assertIsInstance(result, float)

    # Test the expected outputs of the compute method

    def test_compute(self):
        result = self.acc.compute(
            self.arr_a,
            self.arr_b,
            self.statistical_threshold,
            self.feature_labels_a,
            self.feature_labels_b,
        )

        print(result)

        # Here, you would replace 'expected_result' with the actual expected result.
        # This is just a placeholder.
        expected_result = ComputeOutput(
            reliability_a=-0.35206361650271073,
            reliability_b=0.6341959112326657,
            reliability_across=-0.3196747011617917,
            corrected_correlation=np.nan,
            pearson=-0.6447863778774613,
            significance_level_a=0.911543526960971,
            significance_level_b=0.03459578128489493,
            pairwise_correlation_a=np.array(
                [
                    [np.nan, -0.52319234, -0.47113574],
                    [-0.52319234, np.nan, -0.01113741],
                    [-0.47113574, -0.01113741, np.nan],
                ]
            ),
            pairwise_correlation_b=np.array(
                [
                    [np.nan, 0.58389025, 0.81495967],
                    [0.58389025, np.nan, 0.4096952],
                    [0.81495967, 0.4096952, np.nan],
                ]
            ),
            pairwise_correlation_across=np.array(
                [
                    [-0.09296517, 0.35073136, -0.59605678],
                    [-0.5887996, -0.21653578, -0.03799203],
                    [-0.17366883, -0.88971143, -0.03773701],
                ]
            ),
        )

        self.assertAlmostEqual(
            result.reliability_a, expected_result.reliability_a, places=7
        )
        self.assertAlmostEqual(
            result.reliability_b, expected_result.reliability_b, places=7
        )
        self.assertAlmostEqual(
            result.reliability_across, expected_result.reliability_across, places=7
        )
        self.assertAlmostEqual(result.pearson, expected_result.pearson, places=7)
        self.assertAlmostEqual(
            result.significance_level_a, expected_result.significance_level_a, places=7
        )
        self.assertAlmostEqual(
            result.significance_level_b, expected_result.significance_level_b, places=7
        )

        print(result.pairwise_correlation_a)
        print(
            expected_result.pairwise_correlation_a,
        )
        assert_array_almost_equal(
            result.pairwise_correlation_a,
            expected_result.pairwise_correlation_a,
            decimal=7,
        )
        assert_array_almost_equal(
            result.pairwise_correlation_b,
            expected_result.pairwise_correlation_b,
            decimal=7,
        )
        assert_array_almost_equal(
            result.pairwise_correlation_across,
            expected_result.pairwise_correlation_across,
            decimal=7,
        )

    def test__reliability(self):

        result = self.acc._reliability(self.arr_a, self.arr_b)

        expected_result = (
            np.array(
                [
                    [np.nan, -0.58072522, -0.51152909],
                    [-0.58072522, np.nan, -0.01113787],
                    [-0.51152909, -0.01113787, np.nan],
                ]
            ),
            np.array(
                [
                    [np.nan, 0.66834512, 1.14162237],
                    [0.66834512, np.nan, 0.43524489],
                    [1.14162237, 0.43524489, np.nan],
                ]
            ),
            np.array(
                [
                    [-0.09323439, 0.36627746, -0.68700852],
                    [-0.67582668, -0.22001859, -0.03801033],
                    [-0.17544712, -1.42053957, -0.03775494],
                ]
            ),
        )

        for res, exp in zip(result, expected_result):
            assert_array_almost_equal(res, exp, decimal=8)

    def test__rac(self):
        result = self.acc._rac(0.5, 0.6, 0.7)

        expected_result = 0.7715167498104596
        self.assertEqual(result, expected_result)

    def test__repeated_mean(self):

        feature_labels_a = np.array([1, 2, 2])
        feature_labels_b = np.array([4, 5, 6, 7, 7])

        result = self.acc._repeated_mean(self.arr_b, feature_labels_a, feature_labels_b)

        expected_result = 0.49723014208345356
        self.assertEqual(result, expected_result)

    def test_ttest(self):
        feature_labels_a = np.array([1, 2, 2])

        result = self.acc.ttest(self.arr_b, feature_labels_a)

        expected_result = 0.04340245858796643
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
