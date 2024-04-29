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


def benchmark_ray_tasks(b: int, h: int, get=True):
    worker = ray.remote(hashloop)
    cpus = int(ray.available_resources()['CPU'])

    start, ends = time(), []
    refs = [worker.remote(b, h) for _ in range(cpus)]

    while refs:
        ready_refs, refs = ray.wait(refs)
        for _ in ready_refs:
            ends.append(time() - start)

        if get: 
            _ = ray.get(ready_refs)

    return ends


def _units(count: int) -> str:
    if 1024**1 <= count < 1024**2: return f"{count/1024**1:.2f}K"
    if 1024**2 <= count < 1024**3: return f"{count/1024**2:.2f}M"
    if 1024**3 <= count < 1024**4: return f"{count/1024**3:.2f}G"
    if 1024**4 <= count < 1024**5: return f"{count/1024**4:.2f}T"
    else:                          return f"{count}"


def experiment(_bytes, _hashes, get): 
    _ = benchmark_ray_tasks(2**22, 2**22, get=True)  # rev it once 
    print("Finished revving!")

    results = np.zeros((len(_bytes), len(_hashes), 224))
    for b_idx, b in enumerate(tqdm(_bytes)): 
        for h_idx, h in enumerate(tqdm(_hashes, leave=False)):
            ends = benchmark_ray_tasks(b, h, get)
            results[b_idx, h_idx] = ends

    return results


def plot_grid(results, _bytes, _hashes):
    rows, cols = results.shape[:2]
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
    plt.suptitle("Each graph shows a sorted list of finish times for 224 processes")

    for x in range(rows):
        for y in range(cols):
            plt.sca(axes[x, y])
            plt.plot(results[x, y])
            axes[x, y].set_title(f"{_units(_bytes[x])}B, {_hashes[y]} hashes")
            if y == 0: axes[x, y].set_ylabel("Time [s]")
            if x == rows - 1: axes[x, y].set_xlabel("Ray task [#]")

    plt.show()


def plot_two_grids(results1, results2, _bytes, _hashes):
    rows, cols = results1.shape[:2]
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
    plt.suptitle("Each graph shows a sorted list of finish times for 224 processes")

    for x in range(rows):
        for y in range(cols):
            plt.sca(axes[x, y])
            plt.plot(results1[x, y], label="Get")
            plt.plot(results2[x, y], label="No get")
            axes[x, y].set_title(f"{_units(_bytes[x])}B, {_hashes[y]} hashes")
            plt.legend()
            if y == 0: axes[x, y].set_ylabel("Time [s]")
            if x == rows - 1: axes[x, y].set_xlabel("Ray task [#]")

    plt.show()

    
if __name__ == "__main__":
    # NOTE: Delete the savefile to force a rerun of the experiments
    savefile = "npz/timelines.npz"
    try:
        timelines = np.load(savefile)
        results_get   = timelines['arr_0']
        results_noget = timelines['arr_1']
        _bytes        = timelines['arr_2']
        _hashes       = timelines['arr_3']
    except: 
        _bytes  = [2**x for x in [19, 20, 21, 22, 23, 24]]
        _hashes = [2**x for x in [12, 19, 20]]
        results_get = experiment(_bytes, _hashes, get=True)
        results_noget = experiment(_bytes, _hashes, get=False)
        np.savez(savefile, results_get, results_noget, _bytes, _hashes)

    # plot_grid(results, _bytes, _hashes)
    plot_two_grids(results_get, results_noget, _bytes, _hashes)

