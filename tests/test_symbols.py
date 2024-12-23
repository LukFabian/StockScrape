import unittest
from unittest import TestCase

from scrape.symbols import format_to_bigint


class Test(TestCase):
    def test_valid_float_input(self):
        self.assertEqual(format_to_bigint(123.45), "12345")
        self.assertEqual(format_to_bigint(0.01), "1")
        self.assertEqual(format_to_bigint(12.0), "1200")
        self.assertEqual(format_to_bigint(99.999), "10000")  # Rounding up

    def test_zero_input(self):
        self.assertEqual(format_to_bigint(0.0), "0")

    def test_negative_input(self):
        self.assertEqual(format_to_bigint(-123.45), "-12345")
        self.assertEqual(format_to_bigint(-0.01), "-1")
        self.assertEqual(format_to_bigint(-12.0), "-1200")
        self.assertEqual(format_to_bigint(-99.999), "-10000")  # Rounding up

    def test_large_float(self):
        self.assertEqual(format_to_bigint(12345678.90), "1234567890")
        self.assertEqual(format_to_bigint(-9876543.21), "-987654321")

    def test_edge_case_rounding(self):
        self.assertEqual(format_to_bigint(0.005), "0")  # Rounding down, as 0 is even
        self.assertEqual(format_to_bigint(1.004), "100")  # No rounding up
        self.assertEqual(format_to_bigint(2.995), "300")  # Rounding up

    def test_integer_input(self):
        self.assertEqual(format_to_bigint(100), "10000")  # 100.00 -> 10000
        self.assertEqual(format_to_bigint(-50), "-5000")  # -50.00 -> -5000
