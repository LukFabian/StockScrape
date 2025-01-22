from typing import List, Tuple
import statistics


def calculate_adx(high: List[int], low: List[int], close: List[int], period: int) -> List[float]:
    """
    Calculate the Average Directional Movement Index (ADX).

    :param high: List of high prices.
    :param low: List of low prices.
    :param close: List of close prices.
    :param period: The period over which to calculate the ADX.
    :return: List of ADX values.
    """
    # Ensure the lists have the same length
    if not (len(high) == len(low) == len(close)):
        raise ValueError("Input lists must have the same length.")

    if period <= 0:
        raise ValueError("Period must be an integer > 0")
    true_range = calculate_true_range(high, low, close)
    average_true_range = calculate_average_true_range(true_range, period)
    smooth_positive_directional_indexes, smooth_negative_directional_indexes = calculate_smoothed_directional_indexes(
        high, low, period)
    positive_directional_movement_indexes, negative_directional_movement_indexes = calculate_directional_moving_indexes(
        average_true_range, smooth_positive_directional_indexes, smooth_negative_directional_indexes)
    directional_index = calculate_directional_index(positive_directional_movement_indexes,
                                                    negative_directional_movement_indexes)
    adx = calculate_average_directional_index(directional_index, period)
    return adx


def calculate_true_range(high: List[int], low: List[int], close: List[int]) -> List[int]:
    """
    Calculate the True Range

    :param high: List of high prices.
    :param low: List of low prices.
    :param close: List of close prices.
    :return: List of True Range values.
    """
    # Ensure the lists have the same length
    if not (len(high) == len(low) == len(close)):
        raise ValueError("Input lists must have the same length.")
    true_ranges = list()
    for i in range(1, len(high)):
        high_minus_low = high[i] - low[i]
        high_minus_previous_close = abs(high[i] - close[i - 1])
        low_minus_previous_close = abs(low[i] - close[i - 1])
        true_range = max(high_minus_low, high_minus_previous_close, low_minus_previous_close)
        true_ranges.append(true_range)
    return true_ranges


def calculate_average_true_range(true_range: List[int], period: int) -> List[float]:
    """
    Calculate the Average True Range

    :param true_range: List of true range values
    :param period: The period over which to calculate the ATR.
    :return: List of Average True Range values.
    """
    # Ensure the lists have the same length
    if not (len(true_range) >= period):
        raise ValueError("True Range list has to have a length longer or equal to the period")
    average_true_range = list()
    average_true_range.append(statistics.fmean(true_range[0:period]))
    for i in range(period, len(true_range)):
        average_true_range.append((average_true_range[i - period] * (period - 1) + true_range[i]) / period)
    return average_true_range


def calculate_smoothed_directional_indexes(high: List[int], low: List[int], period: int) -> Tuple[
    List[float], List[float]]:
    """
    Calculate the Smoothed Directional Movement Indexes

    :param period: The period over which to smooth the Directional Movement Indexes.
    :param high: List of high prices.
    :param low: List of low prices.
    :return: Positive and Negative Smoothed Directional Movement Indexes.
    """
    # Ensure the lists have the same length
    if not (len(high) == len(low) and len(low) >= period):
        raise ValueError(
            "High and Low lists have to have equal length and this length must be greater or equal to period")
    positive_dx_list = list()
    negative_dx_list = list()
    for i in range(1, len(high)):
        high_minus_previous_high = high[i] - high[i - 1]
        previous_low_minus_low = low[i - 1] - low[i]
        positive_dx = high_minus_previous_high if high_minus_previous_high > previous_low_minus_low and high_minus_previous_high > 0 else 0
        negative_dx = previous_low_minus_low if previous_low_minus_low > high_minus_previous_high and previous_low_minus_low > 0 else 0
        positive_dx_list.append(positive_dx)
        negative_dx_list.append(negative_dx)
    smooth_positive_dx = list()
    smooth_negative_dx = list()
    for i in range(period - 1, len(positive_dx_list)):
        smooth_positive_dx.append(statistics.fmean(positive_dx_list[i - period + 1: i+1]))
        smooth_negative_dx.append(statistics.fmean(negative_dx_list[i - period + 1:i+1]))
    return smooth_positive_dx, smooth_negative_dx


def calculate_directional_moving_indexes(average_true_range: List[float], positive_directional_index: List[float],
                                         negative_directional_index: List[float]) -> Tuple[List[float], List[float]]:
    """
    Calculate the Directional Moving Indexes

    :param average_true_range: List of average true range values.
    :param positive_directional_index: List of positive directional index values.
    :param negative_directional_index: List of negative directional index values.
    :return: Positive and Negative Directional Moving Indexes.
    """
    if not len(average_true_range) == len(positive_directional_index) == len(negative_directional_index):
        raise ValueError("Input lists must have the same length.")
    positive_directional_moving_indexes = list()
    negative_directional_moving_indexes = list()
    for i in range(len(average_true_range)):
        positive_directional_moving_indexes.append((positive_directional_index[i] / average_true_range[i]) * 100)
        negative_directional_moving_indexes.append((negative_directional_index[i] / average_true_range[i]) * 100)
    return positive_directional_moving_indexes, negative_directional_moving_indexes


def calculate_directional_index(positive_directional_movement_index: List[float],
                                negative_directional_movement_index: List[float]) -> List[float]:
    """
    Calculate the Directional Index

    :param positive_directional_movement_index: List of positive directional movement index values.
    :param negative_directional_movement_index: List of negative directional movement index values.
    :return: List of Directional Index values.
    """
    if not len(positive_directional_movement_index) == len(negative_directional_movement_index):
        raise ValueError("Input lists must have the same length.")
    directional_index = list()
    for i in range(len(positive_directional_movement_index)):
        directional_index.append(
            (abs(positive_directional_movement_index[i] - negative_directional_movement_index[i]) /
             (positive_directional_movement_index[i] + negative_directional_movement_index[i])) * 100)
    return directional_index


def calculate_average_directional_index(directional_index: List[float], period: int) -> List[float]:
    """
    Calculate the Average Directional Index (ADX).

    :param directional_index: List of directional index values.
    :param period: The period over which to calculate the ADX.
    :return: List of Average Directional Index values.
    :raises ValueError: If the directional index list length is less than the specified period.
    """
    if len(directional_index) < period:
        raise ValueError("directional index list must have a length equal to or greater than period")
    adx = [statistics.fmean(directional_index[0:period])]
    for i in range(period, len(directional_index)):
        adx.append(((adx[i - period] * (period - 1)) + directional_index[i]) / period)
    return adx
