import unittest
import selector
import numpy as np


from itertools import pairwise



class Test_Selector(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.array_1 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        self.array_2 = np.array([np.nan, np.nan, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        self.array_3 = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
        self.intervals_1 = np.array([1, 2, 3])
        self.size_1 = 2
        self.size_2 = 4

    def test__interval_choice__no_nans(self):
        selection = selector.interval_choice(self.array_1, self.size_1, self.intervals_1)

        out_of_interval = self.array_1[self.array_1 >= self.intervals_1[-1]]
        for element in out_of_interval:
            self.assertNotIn(element, selection)

        self.assertEqual(len(selection), len(list(pairwise(self.intervals_1))) * self.size_1)

    def test__interval_choice__no_replace(self):
        with self.assertRaises(ValueError) as error:
            selector.interval_choice(self.array_1, self.size_2, self.intervals_1, replace = False)

        self.assertIsInstance(error.exception, ValueError)

    def test__interval_choice__with_nans(self):
        selection = selector.interval_choice(self.array_2, self.size_1, self.intervals_1)
        
        self.assertNotIn(np.nan, selection)

    def test__arginterval_choice__only_4_elements(self):
        selection = selector.arginterval_choice(self.array_3, self.size_1, self.intervals_1, replace = False)

        for index in range(selection.size):
            self.assertIn(index, selection)