import sys
import unittest

from src.utils.geometry import point_in_polygon


class GeometryTests(unittest.TestCase):
    def test_point_in_polygon(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertTrue(point_in_polygon((5, 5), square))
        self.assertFalse(point_in_polygon((15, 5), square))


if __name__ == "__main__":
    unittest.main()
