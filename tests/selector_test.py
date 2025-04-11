import unittest
import selector
import numpy as np


from itertools import pairwise



class Test_Selector(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.array_1 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        self.intervals_1 = np.array([1, 2, 3])
        self.size_1 = 2

    def test_interval_choice(self):
        result = selector.interval_choice(self.array_1, self.size_1, self.intervals_1)


        out_of_interval = self.array_1[self.array_1 > self.intervals_1[-1]]
        for element in out_of_interval:
            self.assertNotIn(element, result)

        self.assertEquals(len(result), len(list(pairwise(self.intervals_1))) * self.size_1)

if __name__ == '__main__':
    unittest.main()