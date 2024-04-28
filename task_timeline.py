"""
Starts a large number of computation and communication-heavy jobs, 
and plots when they start and return.

Should show that increasing the amount of work only moves the graphs up and
down, while increasing the amount of data they send back changes the slope of
when they finish.

Shows that Ray utilizes the network relatively well (~50% of max bandwidth).
"""
from os import urandom
from time import time, sleep
from hashlib import sha512, md5, blake2b
from multiprocessing import cpu_count, Pool

import numpy as np
import ray
from tqdm import tqdm
import matplotlib.pyplot as plt


ray.init("ray://clusterfuzz.boolsi.com:10001")


# def hashloop(b: int, hashes: int, hash=blake2b, chunk=64):
#     """
#     Iterates over the given `b`-byte array, running a given `hash` function
#     `hashes` times over some random data and storing the results in the array.
#     """
#     array = bytearray(b)
# 
#     seed = urandom(chunk)
#     for idx in range(b // chunk):
#         for _ in range(hashes):
#             seed = hash(seed).digest()
# 
#         array[idx*chunk:idx*chunk+chunk] = seed
# 
#     return array


def hashloop(b: int, c: int):
    """
    Performs `c` iterations of some hash function, and returns `b` bytes.
    Goal is to freely change comptutational and network requirements of the worker,
    unlike in the previous implementation.
    """
    array = bytearray(b)

    seed = urandom(64)
    for _ in range(c):
        seed = blake2b(seed).digest()

    return array


def benchmark_ray_tasks_with_getting(b: int, h: int):
    worker = ray.remote(hashloop)
    cpus = int(ray.available_resources()['CPU'])

    start, ends = time(), []
    refs = [worker.remote(b, h) for _ in range(cpus)]

    while refs:
        ready_refs, refs = ray.wait(refs)
        for _ in ready_refs:
            ends.append(time() - start)

        _ = ray.get(ready_refs)

    return ends


def _byte_units(count: int) -> str:
    if 1024**1 <= count < 1024**2:
        return f"{count/1024:.2f}KB"
    if 1024**2 <= count < 1024**3:
        return f"{count/1024**2:.2f}MB"
    if 1024**3 <= count < 1024**4:
        return f"{count/1024**3:.2f}GB"
    if 1024**4 <= count < 1024**5:
        return f"{count/1024**4:.2f}TB"
    else:
        return f"{count}B"


def experiment(_bytes, _hashes): 
    _ = benchmark_ray_tasks_with_getting(2**22, 2**22)  # rev it once 
    print("Finished revving!")

    results = np.zeros((len(_bytes), len(_hashes), 224))
    for b_idx, b in enumerate(tqdm(_bytes)): 
        for h_idx, h in enumerate(tqdm(_hashes, leave=False)):
            ends = benchmark_ray_tasks_with_getting(b, h)
            results[b_idx, h_idx] = ends
            plt.plot(ends, label=f"{h} hashes, {_byte_units(b)}, over 244 cores")

    plt.legend()
    plt.xlabel("Ray task [#]")
    plt.ylabel("Time [s]")
    plt.show()

    return results


def plot_grid(results, _bytes, _hashes):
    rows, cols = results.shape[:2]
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
    plt.suptitle("Each graph shows a sorted list of finish times for 224 processes")

    for x in range(rows):
        for y in range(cols):
            plt.sca(axes[x, y])
            plt.plot(results[x, y])
            axes[x, y].set_title(f"{_byte_units(_bytes[x])}, {_hashes[y]} hashes")
            if y == 0: axes[x, y].set_ylabel("Time [s]")
            if x == rows - 1: axes[x, y].set_xlabel("Ray task [#]")


    plt.show()

    
if __name__ == "__main__":
    _bytes  = [2**x for x in [12, 14, 16, 18, 20]]
    _hashes = [2**x for x in [12, 19, 20]]

    #
    # NOTE: uncomment these 2 lines to run the experiments, 
    #       but also comment out the 4 lines below
    #
    # results = experiment(_bytes, _hashes)
    # np.savez("timelines.npz", results, _bytes, _hashes)

    timelines = np.load("timelines.npz")
    results = timelines['arr_0']
    _bytes  = timelines['arr_1']
    _hashes = timelines['arr_2']

    plot_grid(results, _bytes, _hashes)
