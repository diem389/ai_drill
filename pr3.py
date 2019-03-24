import pathos.multiprocessing as mp2
import math


if __name__ == '__main__':
    p = mp2.Pool()

    r = p.map(lambda x: x**2, range(10))
    print(r)
