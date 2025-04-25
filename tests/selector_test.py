import unittest
import sensingpy.selector as selector
import numpy as np
from itertools import pairwise

class Test_Selector(unittest.TestCase):
    def setUp(self):
        """Set up test arrays and parameters used across multiple tests."""
        # Arrays with different characteristics
        self.array_1 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])  # Regular array
        self.array_2 = np.array([np.nan, np.nan, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])  # Array with NaNs
        self.array_3 = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4])  # Smaller array
        
        # Test parameters
        self.intervals_1 = np.array([1, 2, 3])
        self.size_1 = 2
        self.size_2 = 4

    def test_interval_choice_basic(self):
        """Test basic interval_choice functionality with regular array."""
        selection = selector.interval_choice(self.array_1, self.size_1, self.intervals_1)
        
        # Verify selection size
        expected_size = len(list(pairwise(self.intervals_1))) * self.size_1
        self.assertEqual(len(selection), expected_size)
        
        # Verify values are within intervals
        for val in selection:
            self.assertTrue(any(val >= start and val < end 
                              for start, end in pairwise(self.intervals_1)))

    def test_interval_choice_no_replace(self):
        """Test interval_choice raises error when sample size exceeds available values."""
        with self.assertRaises(ValueError) as error:
            selector.interval_choice(self.array_1, self.size_2, self.intervals_1, replace=False)
        self.assertIn("Cannot take a larger sample than population", str(error.exception))

    def test_interval_choice_with_nans(self):
        """Test interval_choice properly handles arrays containing NaN values."""
        selection = selector.interval_choice(self.array_2, self.size_1, self.intervals_1)
        
        # Verify no NaNs in selection
        self.assertTrue(np.all(~np.isnan(selection)))
        
        # Verify selection size
        expected_size = len(list(pairwise(self.intervals_1))) * self.size_1
        self.assertEqual(len(selection), expected_size)

    def test_arginterval_choice_basic(self):
        """Test basic arginterval_choice functionality."""
        selection = selector.arginterval_choice(self.array_3, self.size_1, self.intervals_1)
        
        # Verify indices are valid
        self.assertTrue(np.all(selection >= 0))
        self.assertTrue(np.all(selection < len(self.array_3)))
        
        # Verify selected values are within intervals
        selected_values = self.array_3[selection]
        for val in selected_values:
            self.assertTrue(any(val >= start and val < end 
                              for start, end in pairwise(self.intervals_1)))

    def test_arginterval_choice_no_replace(self):
        """Test arginterval_choice with replacement disabled."""
        selection = selector.arginterval_choice(
            self.array_3, self.size_1, self.intervals_1, replace=False)
        
        # Verify no duplicate indices when replace=False
        self.assertEqual(len(selection), len(set(selection)))

    def test_empty_array(self):
        """Test behavior with empty input array."""
        empty_array = np.array([])
        with self.assertRaises(ValueError):
            selector.interval_choice(empty_array, self.size_1, self.intervals_1)
        with self.assertRaises(ValueError):
            selector.arginterval_choice(empty_array, self.size_1, self.intervals_1)

    def test_invalid_intervals(self):
        """Test behavior with invalid intervals."""
        invalid_intervals = np.array([2, 1])  # Decreasing intervals
        with self.assertRaises(ValueError):
            selector.interval_choice(self.array_1, self.size_1, invalid_intervals)

if __name__ == '__main__':
    unittest.main()