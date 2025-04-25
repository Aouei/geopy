import unittest
import sensingpy.masks as masks
import numpy as np


class Test_Masks(unittest.TestCase):
    def setUp(self):
        """Set up test arrays that will be used across multiple tests."""
        self.array_with_nans = np.array([np.nan, 1, 2, 3])
        self.simple_array = np.array([1, 2, 3, 4, 5])
        self.test_value = 3
        
    def test_is_valid(self):
        """Test is_valid mask identifies non-NaN values correctly."""
        result = masks.is_valid(self.array_with_nans)
        self.assertEqual(np.count_nonzero(result), 3)
        self.assertTrue(np.array_equal(result, [False, True, True, True]))

    def test_is_lt(self):
        """Test is_lt mask identifies values less than threshold."""
        result = masks.is_lt(self.simple_array, self.test_value)
        self.assertTrue(np.array_equal(result, [True, True, False, False, False]))

    def test_is_eq(self):
        """Test is_eq mask identifies values equal to threshold."""
        result = masks.is_eq(self.simple_array, self.test_value)
        self.assertTrue(np.array_equal(result, [False, False, True, False, False]))

    def test_is_gt(self):
        """Test is_gt mask identifies values greater than threshold."""
        result = masks.is_gt(self.simple_array, self.test_value)
        self.assertTrue(np.array_equal(result, [False, False, False, True, True]))

    def test_is_lte(self):
        """Test is_lte mask identifies values less than or equal to threshold."""
        result = masks.is_lte(self.simple_array, self.test_value)
        self.assertTrue(np.array_equal(result, [True, True, True, False, False]))

    def test_is_gte(self):
        """Test is_gte mask identifies values greater than or equal to threshold."""
        result = masks.is_gte(self.simple_array, self.test_value)
        self.assertTrue(np.array_equal(result, [False, False, True, True, True]))

    def test_is_in_range(self):
        """Test is_in_range mask identifies values within given range."""
        vmin, vmax = 2, 4
        result = masks.is_in_range(self.simple_array, vmin, vmax)
        self.assertTrue(np.array_equal(result, [False, True, True, True, False]))

    def test_with_empty_array(self):
        """Test all masks handle empty arrays correctly."""
        empty_array = np.array([])
        self.assertEqual(len(masks.is_valid(empty_array)), 0)
        self.assertEqual(len(masks.is_lt(empty_array, 1)), 0)
        self.assertEqual(len(masks.is_eq(empty_array, 1)), 0)
        self.assertEqual(len(masks.is_gt(empty_array, 1)), 0)
        self.assertEqual(len(masks.is_lte(empty_array, 1)), 0)
        self.assertEqual(len(masks.is_gte(empty_array, 1)), 0)
        self.assertEqual(len(masks.is_in_range(empty_array, 1, 2)), 0)


if __name__ == '__main__':
    unittest.main()