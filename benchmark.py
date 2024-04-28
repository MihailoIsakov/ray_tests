from os import urandom
from time import time, sleep
from hashlib import sha512, md5, blake2b
from multiprocessing import cpu_count, Pool

import numpy as np
import ray
from tqdm import tqdm


def hashloop(b: int, hashes: int, hash=blake2b, chunk=64):
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
    runtimes = np.zeros((len(bytes_list), len(hashes_list)))

    for h_idx, h in enumerate(tqdm(hashes_list)):
        for b_idx, b in enumerate(tqdm(bytes_list, leave=False)):
            start = time()

            cpus = cpu_count()
            with Pool() as pool:
                pool.map(worker, [(b // cpus, h) for _ in range(cpus)])

            runtimes[b_idx, h_idx] = time() - start

    return runtimes


def benchmark_ray_pool(bytes_list: list[int], hashes_list: list[int]) -> np.ndarray:
    from ray.util.multiprocessing import Pool
    runtimes = np.zeros((len(bytes_list), len(hashes_list)))

    ray.init("ray://clusterfuzz.boolsi.com:10001")
    pool = Pool(224, ray_remote_args={"num_cpus": 1})
    # NOTE: this throws an error if a pool is created?
    # cpus = int(ray.available_resources()['CPU'])
    cpus = len(pool._actor_pool)
    print(f"Ray CPU count: {cpus}")

    for h_idx, h in enumerate(tqdm(hashes_list)):
        for b_idx, b in enumerate(tqdm(bytes_list, leave=False)):
            start = time()

            pool.map(worker, [(b // cpus, h) for _ in range(cpus)])

            runtimes[b_idx, h_idx] = time() - start

    return runtimes


def benchmark_ray_task(bytes_list: list[int], hashes_list: list[int]) -> np.ndarray:
    runtimes = np.zeros((len(bytes_list), len(hashes_list)))
    worker = ray.remote(hashloop)

    ray.init("ray://clusterfuzz.boolsi.com:10001")
    cpus = int(ray.available_resources()['CPU'])
    print(f"Ray CPU count: {cpus}")

    # FIXME: a warmup seems necessary, otherwise early results are large
    # however, sleeping nor waiting around doesn't help
    # start = time()
    # while time() - start < 5:
    #     sleep(0.001)

    for h_idx, h in enumerate(tqdm(hashes_list)):
        for b_idx, b in enumerate(tqdm(bytes_list, leave=False)):
            start = time()

            workers = [worker.remote(b // cpus, h) for _ in range(cpus)]
            _ = ray.get(workers)

            runtimes[b_idx, h_idx] = time() - start

    return runtimes


def benchmark_ray_task_wait(bytes_list: list[int], hashes_list: list[int]) -> np.ndarray:
    runtimes = np.zeros((len(bytes_list), len(hashes_list)))
    worker = ray.remote(hashloop)

    ray.init("ray://clusterfuzz.boolsi.com:10001")
    cpus = int(ray.available_resources()['CPU'])
    print(f"Ray CPU count: {cpus}")

    for h_idx, h in enumerate(tqdm(hashes_list)):
        for b_idx, b in enumerate(tqdm(bytes_list, leave=False)):
            start = time()

            refs = [worker.remote(b // cpus, h) for _ in range(cpus)]
            while refs:
                ready_refs, refs = ray.wait(refs)
                _ = ray.get(ready_refs)

            runtimes[b_idx, h_idx] = time() - start

    return runtimes


if __name__ == "__main__":
    sqrt2 = 2**0.5
    bytes_list = [int(sqrt2 ** x) for x in range(40, 56)]
    hashes_list = [2**x for x in range(7, 11)]
    # bytes_list = [int(sqrt2**55)]
    # hashes_list = [2**10]

    print(f"Bytes list: {bytes_list}")
    print(f"Hash operations list: {hashes_list}")

    # runtimes = benchmark_single_thread(bytes_list, hashes_list)
    # print("Single thread runtimes:")
    # print(runtimes)

    # runtimes = benchmark_multiprocessing_pool(bytes_list, hashes_list)
    # print("Multiprocessing pool runtimes:")
    # print(runtimes)

    # runtimes = benchmark_ray_pool(bytes_list, hashes_list)
    # print("Ray pool runtimes:")
    # print(runtimes)

    # runtimes = benchmark_ray_task(bytes_list, hashes_list)
    # print("Ray task runtimes:")
    # print(runtimes)

    runtimes = benchmark_ray_task_wait(bytes_list, hashes_list)
    print("Ray task-wait runtimes:")
    print(runtimes)

    # plot_runtimes(runtimes, bytes_list, hashes_list)
    # plot_runtimes((runtimes.T / bytes_list).T, bytes_list, hashes_list)
