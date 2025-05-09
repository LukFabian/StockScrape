import unittest
from financial_mathematics.relative_strength_index import calculate_average_movement, calculate_average_downward_movement, calculate_relative_strength


class TestRelativeStrength(unittest.TestCase):

    def test_calculate_average_movement(self):
        close = [100, 105, 102, 108]
        expected = [5, 0, 6]
        result = calculate_average_movement(close)
        self.assertEqual(result, expected)

    def test_calculate_average_downward_movement(self):
        close = [100, 95, 98, 94]
        expected = [5, 0, 4]
        result = calculate_average_downward_movement(close)
        self.assertEqual(result, expected)

    def test_calculate_relative_strength_basic(self):
        close = [10000, 10100, 10200, 10100, 10300, 10400, 10300, 10500, 10600, 10700, 10800, 10900, 11000, 11100]
        period = 5
        result = calculate_relative_strength(close, period)
        # Check length
        self.assertEqual(len(result), len(close) - 1 - (period - 1))
        # Values should be between 0 and 100
        for rsi in result:
            self.assertGreaterEqual(rsi, 0)
            self.assertLessEqual(rsi, 100)

    def test_calculate_relative_strength_constant_prices(self):
        close = [10000] * 10
        period = 5
        result = calculate_relative_strength(close, period)
        for rsi in result:
            self.assertEqual(rsi, 0.0)

    def test_calculate_relative_strength_only_up(self):
        close = [10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700]
        period = 3
        result = calculate_relative_strength(close, period)
        for rsi in result:
            self.assertGreaterEqual(rsi, 99)
            self.assertLessEqual(rsi, 100)

    def test_calculate_relative_strength_only_down(self):
        close = [10700, 10600, 10500, 10400, 10300, 10200, 10100, 10000]
        period = 3
        result = calculate_relative_strength(close, period)
        for rsi in result:
            self.assertGreaterEqual(rsi, 0)
            self.assertLessEqual(rsi, 1)


if __name__ == '__main__':
    unittest.main()