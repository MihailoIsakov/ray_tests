from multiprocessing import cpu_count
from tqdm import tqdm
from time import time
from work import hashloop
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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

    pool = Pool()
    for b_idx, b in enumerate(tqdm(bytes_list, leave=False)):
        for h_idx, h in enumerate(tqdm(hashes_list)):
            start = time()

            cpus = cpu_count()
            pool.map(worker, [(b // cpus, h) for _ in range(cpus)])

            runtimes[b_idx, h_idx] = time() - start

    return runtimes


def plot_runtimes(runtimes: np.ndarray, bytes_list: list[int], hashes_list: list[int]):
    plt.plot(runtimes, label=[f"{h} hashes" for h in hashes_list])
    plt.xticks(range(len(bytes_list)), [f"{b} bytes" for b in bytes_list])
    plt.show()


if __name__ == "__main__":
    sqrt2 = 2**0.5
    bytes_list = [int(sqrt2 ** x) for x in range(40, 50)]
    hashes_list = [int(sqrt2 ** x) for x in range(0, 14)]

    # runtimes = benchmark_single_thread(bytes_list, hashes_list)
    # print("Single thread runtimes:")
    # print(runtimes)
    # runtimes = benchmark_multiprocessing_pool(bytes_list, hashes_list)
    runtimes = benchmark_ray_pool_local(bytes_list, hashes_list)
    print("Multiprocessing pool runtimes:")
    print(runtimes)
    plot_runtimes(runtimes, bytes_list, hashes_list)
    plot_runtimes((runtimes.T / bytes_list).T, bytes_list, hashes_list)
