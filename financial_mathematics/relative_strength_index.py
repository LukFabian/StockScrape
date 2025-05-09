import statistics
from typing import List


def calculate_average_movement(close: List[int]) -> List[int]:
    upward_movement = []
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            upward_movement.append(close[i] - close[i - 1])
        else:
            upward_movement.append(0)
    return upward_movement


def calculate_average_downward_movement(close: List[int]) -> List[int]:
    downward_movement = []
    for i in range(1, len(close)):
        if close[i] < close[i - 1]:
            downward_movement.append(close[i - 1] - close[i])
        else:
            downward_movement.append(0)
    return downward_movement


def calculate_relative_strength(close: List[int], period: int) -> List[float]:
    downward_movement = calculate_average_downward_movement(close)
    upward_movement = calculate_average_movement(close)

    avg_downward_movement = [statistics.fmean(downward_movement[0:period])]
    avg_upward_movement = [statistics.fmean(upward_movement[0:period])]

    for i in range(period, len(downward_movement)):
        avg_up = (avg_upward_movement[-1] * (period - 1) + upward_movement[i]) / period
        avg_down = (avg_downward_movement[-1] * (period - 1) + downward_movement[i]) / period
        avg_upward_movement.append(avg_up)
        avg_downward_movement.append(avg_down)

    relative_strength_index = []
    for up, down in zip(avg_upward_movement, avg_downward_movement):
        if down == 0 and up == 0:
            rsi = 0.0  # flat prices → RSI 0
        elif down == 0:
            rsi = 100.0  # only gains → RSI 100
        else:
            rs = up / down
            rsi = 100 - (100 / (1 + rs))
        relative_strength_index.append(rsi)

    return relative_strength_index
