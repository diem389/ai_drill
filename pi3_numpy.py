import numpy as np
import time


def estimate_points_num_in_quarter_circle(sample_num):
    np.random.seed()

    xs = np.random.uniform(0, 1, sample_num)
    ys = np.random.uniform(0, 1, sample_num)
    is_in_circle = (xs * xs + ys * ys) <= 1
    return np.sum(is_in_circle)


if __name__ == "__main__":

    sample_num = 10**8
    t1 = time.time()
    in_circle_num = estimate_points_num_in_quarter_circle(sample_num)
    pi = float(in_circle_num) / sample_num * 4
    print("Estimated PI:", pi)
    print("Delta: ", time.time() - t1)
