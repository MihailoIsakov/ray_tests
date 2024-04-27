from multiprocessing import cpu_count
from tqdm import tqdm
from time import time
import numpy as np
import matplotlib.pyplot as plt
import ray

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


def worker(args):
    s, b = args
    _ = hashloop(s, b)


def benchmark_single_thread(bytes_list: list[int], hashes_list: list[int]) -> np.ndarray:
    runtimes = np.zeros((len(bytes_list), len(hashes_list)))

    for b_idx, b in enumerate(tqdm(bytes_list, leave=False)):
        for h_idx, h in enumerate(tqdm(hashes_list)):
            start = time()
            _ = hashloop(b, h)
            runtimes[b_idx, h_idx] = time() - start

    return runtimes


def benchmark_multiprocessing_pool(bytes_list: list[int], hashes_list: list[int]) -> np.ndarray:
    from multiprocessing import Pool
    runtimes = np.zeros((len(bytes_list), len(hashes_list)))

    for h_idx, h in enumerate(tqdm(hashes_list)):
        for b_idx, b in enumerate(tqdm(bytes_list, leave=False)):
            start = time()

            cpus = cpu_count()
            with Pool() as pool:
                pool.map(worker, [(b // cpus, h) for _ in range(cpus)])

            runtimes[b_idx, h_idx] = time() - start

    return runtimes


def benchmark_ray_pool_local(bytes_list: list[int], hashes_list: list[int]) -> np.ndarray:
    from ray.util.multiprocessing import Pool
    runtimes = np.zeros((len(bytes_list), len(hashes_list)))

    ray.init("ray://clusterfuzz.boolsi.com:10001")
    pool = Pool()
    cpus = int(ray.available_resources()['CPU'])
    print(f"Ray CPU count: {cpus}")
    cpus *= 8

    for h_idx, h in enumerate(tqdm(hashes_list)):
        for b_idx, b in enumerate(tqdm(bytes_list, leave=False)):
            start = time()

            pool.map(worker, [(b // cpus, h) for _ in range(cpus)])

            runtimes[b_idx, h_idx] = time() - start

    return runtimes


def plot_runtimes(runtimes: np.ndarray, bytes_list: list[int], hashes_list: list[int]):
    plt.plot(runtimes, label=[f"{h} hashes" for h in hashes_list])
    plt.xticks(range(len(bytes_list)), [f"{b} bytes" for b in bytes_list])
    plt.show()


if __name__ == "__main__":
    sqrt2 = 2**0.5
    bytes_list = [int(sqrt2 ** x) for x in range(40, 56)]
    hashes_list = [2**x for x in range(7, 11)]

    print(f"Bytes list: {bytes_list}")
    print(f"Hash operations list: {hashes_list}")

    # runtimes = benchmark_single_thread(bytes_list, hashes_list)
    # print("Single thread runtimes:")
    # print(runtimes)

    runtimes = benchmark_multiprocessing_pool(bytes_list, hashes_list)
    print("Multiprocessing pool runtimes:")
    print(runtimes)

    runtimes = benchmark_ray_pool_local(bytes_list, hashes_list)
    print("Ray pool runtimes:")
    print(runtimes)

    # plot_runtimes(runtimes, bytes_list, hashes_list)
    # plot_runtimes((runtimes.T / bytes_list).T, bytes_list, hashes_list)
