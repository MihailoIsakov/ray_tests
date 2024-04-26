from time import time
from hashlib import sha512
from random import randint
from os import urandom


def hashloop(b: int, hashes: int, hash=sha512, chunk=64):
    """
    Iterates over the given `b`-byte array, running a given `hash` function
    `hashes` times over some random data and storing the results in the array.
    """
    array = bytearray(b)

    seed = urandom(chunk)
    for idx in range(b // chunk):
        for _ in range(hashes):
            seed = hash(seed).digest()

        array[idx*chunk:idx*chunk+chunk] = seed

    return array


