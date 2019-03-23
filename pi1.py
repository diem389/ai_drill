import numpy as np
import random


def estimate_points_in_quarter_circle(no_estimate):

    no_in_circle = 0
    for step in range(no_estimate):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        is_in_unit_circle = x * x + y * y <= 1.0
        no_in_circle += is_in_unit_circle

    return no_in_circle


def try_estimate_pi(no_estimate):
    in_circle_num = estimate_points_in_quarter_circle(no_estimate)
    pi = float(in_circle_num) / no_estimate * 4

    return pi


if __name__ == "__main__":
    no_estimate = 10**7
    print("Estimated PI:", try_estimate_pi(no_estimate))
