import unittest

from financial_mathematics.average_directional_index import calculate_true_range, calculate_average_true_range, \
    calculate_smoothed_directional_indexes, calculate_directional_moving_indexes, calculate_directional_index, \
    calculate_average_directional_index, calculate_adx


class TestCalculateAdx(unittest.TestCase):
    def test_adx(self):
        high = [8564, 8568, 8540, 8534, 8635, 8533, 8482, 8526, 8647, 8770, 8824, 8832, 8821, 8911, 8949, 9028, 9015,
                9064, 8998, 8946, 8717, 8748, 8836, 8503, 8434, 8369, 8390, 8550, 8539, 8479]
        low = [8509, 8511, 8464, 8467, 8520, 8450, 8360, 8394, 8520, 8617, 8732, 8674, 8670, 8755, 8886, 8920, 8946,
               8984, 8866, 8758, 8621, 8619, 8715, 8360, 8234, 8211, 8278, 8439, 8458, 8292]
        close = [8545, 8542, 8490, 8531, 8595, 8457, 8482, 8508, 8592, 8727, 8759, 8746, 8799, 8862, 8946, 9012, 8961,
                 9013, 8907, 8826, 8656, 8718, 8724, 8438, 8256, 8215, 8352, 8493, 8482, 8317]

        period = 14

        expected = [24.5744916941327,
                    23.7160480001222,
                    23.7675419638405]

        result = calculate_adx(high, low, close, period)[0]


class TestCalculateTrueRange(unittest.TestCase):

    def test_calculate_true_range(self):
        # Example inputs
        high = [8564, 8568, 8540, 8534, 8635, 8533, 8482, 8526, 8647, 8770, 8824, 8832, 8821, 8911, 8949]
        low = [8509, 8511, 8464, 8467, 8520, 8450, 8360, 8394, 8520, 8617, 8732, 8674, 8670, 8755, 8886]
        close = [8545, 8542, 8490, 8531, 8595, 8457, 8482, 8508, 8592, 8727, 8759, 8746, 8799, 8862, 8946]

        # Expected output
        expected = [57, 78, 67, 115, 145, 122, 132, 139, 178, 97, 158, 151, 156, 87]

        # Run the function
        result = calculate_true_range(high, low, close)

        for i in range(len(result)):
            self.assertAlmostEqual(expected[i], result[i], places=5)

    def test_invalid_input_lengths(self):
        high = [10100, 10200, 10300]
        low = [9900, 9800]
        close = [10000, 10100, 10200]

        # Expecting a ValueError due to mismatched input lengths
        with self.assertRaises(ValueError):
            calculate_true_range(high, low, close)

    def test_single_element_input(self):
        high = [10100]
        low = [9900]
        close = [10000]

        # True range cannot be calculated with a single element
        expected = []
        result = calculate_true_range(high, low, close)
        self.assertEqual(expected, result)

    def test_negative_values(self):
        high = [-10100, -10200, -10300, -10400]
        low = [-10500, -10600, -10700, -10800]
        close = [-10250, -10350, -10450, -10550]

        # Expected output
        expected = [400, 400, 400]

        # Run the function
        result = calculate_true_range(high, low, close)

        # Assert the result matches expected
        self.assertEqual(expected, result)


class TestCalculateAverageTrueRange(unittest.TestCase):

    def test_calculate_average_true_range(self):
        true_range = [57, 78, 67, 115, 145, 122, 132, 139, 178, 97, 158, 151, 156, 87, 108, 69, 103, 147]
        period = 14

        # Expected output
        expected = [120.142857142857, 119.275510204082, 115.684402332362, 114.778373594336, 117.079918337597]

        # Run the function
        result = calculate_average_true_range(true_range, period)

        # Assert the result matches expected
        for i in range(len(result)):
            self.assertAlmostEqual(expected[i], result[i], places=5)

    def test_invalid_period(self):
        true_range = [57, 78]
        period = 3

        # Expecting a ValueError because the period is greater than the length of the true_range
        with self.assertRaises(ValueError):
            calculate_average_true_range(true_range, period)

    def test_single_period(self):
        true_range = [57, 78, 67]
        period = 1

        # Expected output is the true_range itself when period is 1
        expected = [57.0, 78.0, 67.0]

        # Run the function
        result = calculate_average_true_range(true_range, period)

        # Assert the result matches expected
        for i in range(len(true_range)):
            self.assertAlmostEqual(result[i], expected[i])


class TestCalculateSmoothedDirectionalIndexes(unittest.TestCase):

    def test_calculate_smoothed_directional_indexes(self):
        high = [8564, 8568, 8540, 8534, 8635, 8533, 8482, 8526, 8647, 8770, 8824, 8832, 8821, 8911, 8949]
        low = [8509, 8511, 8464, 8467, 8520, 8450, 8360, 8394, 8520, 8617, 8732, 8674, 8670, 8755, 8886]
        period = 14

        # Expected output
        expected_positive = [41.0714285714286, 43.780612244898]
        expected_negative = [19.2142857142857, 17.8418367346939]

        # Run the function
        result_positive, result_negative = calculate_smoothed_directional_indexes(high, low, period)

        # Assert the results match expected
        for i in range(len(result_positive)):
            self.assertAlmostEqual(result_positive[i], expected_positive[i], places=5)
            self.assertAlmostEqual(result_negative[i], expected_negative[i], places=5)

    def test_invalid_input_lengths(self):
        high = [10100, 10200]
        low = [9900]
        period = 3

        # Expecting a ValueError due to mismatched input lengths
        with self.assertRaises(ValueError):
            calculate_smoothed_directional_indexes(high, low, period)

    def test_small_period(self):
        high = [8564, 8568, 8540, 8534, 8635, 8533, 8482, 8526, 8647, 8770, 8824, 8832, 8821, 8911, 8949]
        low = [8509, 8511, 8464, 8467, 8520, 8450, 8360, 8394, 8520, 8617, 8732, 8674, 8670, 8755, 8886]
        period = 1

        # Expected output
        expected_dx_plus = [4, 0, 0, 101, 0, 0, 44, 121, 123, 54, 0, 0, 90, 38, 79, 0, 49]
        expected_dx_minus = [0, 47, 0, 0, 70, 90, 0, 0, 0, 0, 58, 4, 0, 0, 0, 0, 0]

        # Run the function
        result_positive, result_negative = calculate_smoothed_directional_indexes(high, low, period)

        # Assert the results match expected
        for i in range(len(result_positive)):
            self.assertAlmostEqual(result_positive[i], expected_dx_plus[i], places=5)
            self.assertAlmostEqual(result_negative[i], expected_dx_minus[i], places=5)


class TestCalculateDirectionalMovingIndexes(unittest.TestCase):

    def test_calculate_directional_moving_indexes(self):
        average_true_range = [120.142857142857, 119.275510204082]
        positive_smooth_directional_index = [41.0714285714286, 43.780612244898]
        negative_smooth_directional_index = [19.2142857142857, 17.8418367346939]

        # Expected output
        expected_positive = [34.1854934601665, 36.7054495679699]
        expected_negative = [15.9928656361474, 14.9585079989734]

        # Run the function
        result_positive, result_negative = calculate_directional_moving_indexes(
            average_true_range, positive_smooth_directional_index, negative_smooth_directional_index
        )

        # Assert the results match expected
        for i in range(len(result_positive)):
            self.assertAlmostEqual(result_positive[i], expected_positive[i], places=5)
            self.assertAlmostEqual(result_negative[i], expected_negative[i], places=5)

    def test_invalid_input_lengths(self):
        average_true_range = [50.0, 60.0, 55.0]
        positive_directional_index = [10.0, 15.0]
        negative_directional_index = [8.0, 10.0, 11.0]

        # Expecting a ValueError due to mismatched input lengths
        with self.assertRaises(ValueError):
            calculate_directional_moving_indexes(average_true_range, positive_directional_index,
                                                 negative_directional_index)

    def test_zero_average_true_range(self):
        average_true_range = [50.0, 0.0, 55.0]
        positive_directional_index = [10.0, 15.0, 12.0]
        negative_directional_index = [8.0, 10.0, 11.0]

        # Expecting a ZeroDivisionError
        with self.assertRaises(ZeroDivisionError):
            calculate_directional_moving_indexes(average_true_range, positive_directional_index,
                                                 negative_directional_index)


class TestCalculateDirectionalIndex(unittest.TestCase):

    def test_calculate_directional_index(self):
        positive_directional_movement_index = [34.1854934601665, 36.7054495679699, 35.1416654590818, 35.9384858366109,
                                               32.7154386510438, 29.118735483459]
        negative_directional_movement_index = [15.9928656361474, 14.9585079989734, 14.3212217818913, 13.4032502926751,
                                               19.4002054046483, 23.5830115521303]

        # Expected output
        expected = [36.2559241706161,
                    42.0930617651929,
                    42.0930617651929,
                    45.6717523779232,
                    25.5493978586671,
                    10.5038717740998,
                    5.04171444514452]

        # Run the function
        result = calculate_directional_index(positive_directional_movement_index, negative_directional_movement_index)

        # Assert the results match expected
        for i in range(len(result)):
            self.assertAlmostEqual(result[i], expected[i], places=5)

    def test_invalid_input_lengths(self):
        positive_directional_movement_index = [20.0, 25.0]
        negative_directional_movement_index = [15.0, 20.0, 25.0]

        # Expecting a ValueError due to mismatched input lengths
        with self.assertRaises(ValueError):
            calculate_directional_index(positive_directional_movement_index, negative_directional_movement_index)

    def test_zero_division(self):
        positive_directional_movement_index = [0.0, 0.0, 0.0]
        negative_directional_movement_index = [0.0, 0.0, 0.0]

        # Expecting a ZeroDivisionError due to division by zero
        with self.assertRaises(ZeroDivisionError):
            calculate_directional_index(positive_directional_movement_index, negative_directional_movement_index)


class TestCalculateAverageDirectionalIndex(unittest.TestCase):

    def test_calculate_average_directional_index(self):
        directional_index = [36.2559241706161,
                             42.0930617651929,
                             42.0930617651929,
                             45.6717523779232,
                             25.5493978586671,
                             10.5038717740998,
                             5.04171444514452,
                             1.56011656191984,
                             7.7838144602508,
                             22.9951941754621,
                             30.5732026325752,
                             31.8908068173158,
                             29.4746849355133,
                             12.5562799779849,
                             12.5562799779849]
        period = 14

        # Expected output
        expected = [24.5744916941327, 23.7160480001222]

        # Run the function
        result = calculate_average_directional_index(directional_index, period)

        # Assert the result matches expected
        for i in range(len(result)):
            self.assertAlmostEqual(result[i], expected[i], places=5)

    def test_invalid_period(self):
        directional_index = [25.0, 30.0]
        period = 3

        # Expecting a ValueError because the period is greater than the length of the directional_index
        with self.assertRaises(ValueError):
            calculate_average_directional_index(directional_index, period)

    def test_small_period(self):
        directional_index = [25.0, 30.0, 35.0]
        period = 1

        # Expected output
        expected = [25.0, 30.0, 35.0]

        # Run the function
        result = calculate_average_directional_index(directional_index, period)

        # Assert the result matches expected
        for i in range(len(result)):
            self.assertEqual(result[i], expected[i])
