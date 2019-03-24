from gevent import monkey
monkey.patch_socket()


import gevent
from gevent.coros import Semaphore
import urllib.request
import urllib.error
import urllib.parse

from contextlib import closing
import string
import random


def generate_urls(base_url, num_urls):
    for i range(num_urls):
        yield base_url + "".join(random.sample(string.ascii_lowercase, 10))


def chunked_request(urls, chunk_size=100):
