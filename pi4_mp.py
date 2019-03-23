import time
import argparse
import numpy as np


def estimate_points_num_in_quarter_circle(sample_num):
    np.random.seed()

    xs = np.random.uniform(0, 1, sample_num)
    ys = np.random.uniform(0, 1, sample_num)
    is_in_circle = (xs * xs + ys * ys) <= 1
    return np.sum(is_in_circle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PI estimation with mp')
    parser.add_argument('worker_num', type=int,
                        help='Number of workers e.g. 1, 2, 4')
    parser.add_argument('--total-number', type=int,
                        default=1e8, help='Total sampling number')
    parser.add_argument('--process', action='store_true', default=False,
                        help='True if using process, absent(False) for Threads')

    args = parser.parse_args()
    if args.process:
        print('Using process')
        from multiprocessing import Pool
    else:
        from multiprocessing.dummy import Pool

    total_sample_num = args.total_number
    worker_num = args.worker_num

    pool = Pool()

    sample_num_per_worker = total_sample_num / worker_num
    map_inputs = [int(sample_num_per_worker)] * worker_num
    t1 = time.time()

    results = pool.map(estimate_points_num_in_quarter_circle, map_inputs)
    pool.close()

    pi_estimate = float(sum(results)) / float(total_sample_num) * 4.0
    print("Parallel pi estimate: ", pi_estimate)
    print("Delta: ", time.time() - t1)
