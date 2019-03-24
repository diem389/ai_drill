#import multiprocessing
import os

import pathos.multiprocessing as mp2
#import multiprocessing as mp2
import time


def mainprocess():

    def subprocess(number):
        print('this is the %d th subprocess' % number)

        time.sleep(3)
        print('the %d th subprocess finished' % number)

    print('this is the main process ,process number is : %d' % os.getpid())

    pool = mp2.Pool(3)
    l = range(9)
    pool.map_async(subprocess, l)    # or map_aysnc

    print("hallo")
    print("hallo again")
    pool.close()
    pool.join()


if __name__ == '__main__':
    mainprocess()
