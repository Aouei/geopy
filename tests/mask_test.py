import unittest
import masks
import numpy as np


class Test_Masks(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.array_with_nans = np.array([np.nan, 1, 2, 3])

    def test__is_valid(self):
        result = masks.is_valid(self.array_with_nans)
        self.assertEqual(np.count_nonzero(result), 3)


if __name__ == '__main__':
    unittest.main()