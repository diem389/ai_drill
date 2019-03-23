from multiprocessing import Pool
import pi1
import time


def test_parallel():
    samples = 10 ** 8
    parallel_num = 30

    pool = Pool(processes=parallel_num)

    samples_per_work = samples / parallel_num
    print(samples_per_work)

    trails_per_process = [int(samples_per_work)] * parallel_num

    t1 = time.time()
    num_unit_circle = \
        pool.map(pi1.estimate_points_in_quarter_circle, trails_per_process)

    pi_estimate = float(sum(num_unit_circle)) / float(samples) * 4.0
    print("Parallel pi estimate: ", pi_estimate)
    print("Delta: ", time.time() - t1)


if __name__ == "__main__":
    test_parallel()
