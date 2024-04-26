from time import time
from hashlib import md5
from random import randint


def work(s: float, b: int):
    """Work for a given number of seconds and return a number of bytes."""
    start = time()
    array = bytearray(b)

    # seed first 16 bytes with random values
    for idx in range(16):
        array[idx] = randint(0, 255)

    # seed the rest of the array with the md5 hash of the first 16 bytes
    idx = 16
    while time() - start < s:
        array[idx:idx+16] = md5(array[:idx]).digest()
        idx += 16
        if idx >= b: idx = 0

    return array

