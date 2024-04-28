import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_runtimes(runtimes: np.ndarray, bytes_list: list[int], hashes_list: list[int]):
    plt.plot(runtimes, label=[f"{h} hashes" for h in hashes_list])
    plt.xticks(range(len(bytes_list)), [f"{b} bytes" for b in bytes_list])
    plt.show()


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


def plot_speedups(speedups: np.ndarray, bytes_list: list[int], hashes_list: list[int]):
    sns.heatmap(speedups, annot=True, vmin=0, vmax=4)
    plt.yticks(np.arange(len(bytes_list)) + 0.5, [_byte_units(b) for b in bytes_list], rotation=0)
    plt.xticks(np.arange(len(hashes_list)) + 0.5, [f"{h} iterations" for h in hashes_list])


xeon1_grid = np.array(
         [[ 0.19264579,  0.22354102,  0.31811929,  0.44523978],
          [ 0.22436643,  0.25381327,  0.36069608,  0.55189538],
          [ 0.23292828,  0.31868958,  0.44528794,  0.84482837],
          [ 0.29343247,  0.36009359,  0.55259633,  1.00305653],
          [ 0.3542676 ,  0.43967152,  0.78645658,  1.39617229],
          [ 0.3714509 ,  0.56685376,  0.99304104,  1.86848164],
          [ 0.49450898,  0.73371959,  1.37737417,  2.571455  ],
          [ 0.56145954,  1.00040078,  1.85682654,  3.59270048],
          [ 0.78816581,  1.43504143,  2.53731585,  5.03039694],
          [ 1.02455473,  1.93699574,  3.56573868,  7.0389533 ],
          [ 1.34869003,  2.53282428,  5.00260854,  9.78773117],
          [ 1.8626442 ,  3.57565665,  7.00627971, 13.67528939],
          [ 2.56646609,  5.06412506,  9.78437829, 19.17464423],
          [ 3.5160203 ,  6.90410018, 13.88575888, 27.13183522],
          [ 4.97261381,  9.69180822, 19.06240225, 38.13659716],
          [ 6.96583033, 13.60732365, 27.10958385, 54.05445027]])

ray_pool_grid = np.array(
         [[ 1.68799901, 1.44752932, 1.61567211,  1.63202238],
          [ 1.82576513, 1.66439176, 1.71221471,  1.77965665],
          [ 1.66192889, 1.74002004, 1.7612915 ,  1.6118083 ],
          [ 1.52816129, 1.61156845, 1.73234129,  1.61203885],
          [ 1.62876868, 1.40448117, 1.6093061 ,  1.59858179],
          [ 1.51640105, 1.5916636 , 1.455585  ,  1.74672747],
          [ 1.68249941, 1.49271965, 1.36783075,  1.93571734],
          [ 1.55688548, 1.55886292, 1.47287822,  2.00133848],
          [ 1.29709935, 1.46366405, 1.8333149 ,  2.25727916],
          [ 1.59970236, 1.55190086, 1.99454737,  2.65415812],
          [ 1.45347214, 1.83683395, 2.42913151,  3.35238791],
          [ 1.60490823, 1.92492175, 2.79587483,  4.3140595 ],
          [ 1.67753768, 2.27740598, 3.54298568,  5.57167268],
          [ 1.89844322, 2.8712101 , 4.50481462,  7.6175549 ],
          [ 2.32714272, 3.48587298, 5.79956079, 10.51114988],
          [ 2.6738131 , 4.4882431 , 7.77773547, 14.32401204]])

ray_tasks_grid = np.array(
         [[ 1.79332447,  0.2206912 ,  0.27388239,  0.28678584],
          [ 0.3693459 ,  0.18487906,  0.23296022,  0.30998945],
          [ 0.40075731,  0.22980618,  0.27413249,  0.36585832],
          [ 0.45471859,  0.25981688,  0.34981656,  0.47227883],
          [ 0.45628738,  0.30813766,  0.42087054,  0.6409452 ],
          [ 0.54694033,  0.38261843,  0.51050615,  0.89167786],
          [ 0.56287384,  0.4768405 ,  0.6475749 ,  0.97898078],
          [ 0.64305925,  0.61802435,  0.87083149,  1.24469519],
          [ 1.20867491,  0.79302049,  1.09061933,  1.70244265],
          [ 1.19382715,  1.22543454,  1.60749817,  2.48042035],
          [ 1.33955312,  1.52433014,  2.24756026,  3.39146996],
          [ 1.67570257,  2.06826115,  3.06272054,  4.73969913],
          [ 2.22758341,  2.82403374,  4.13000011,  6.6377418 ],
          [ 3.03568411,  3.95583391,  5.63308668,  9.17397523],
          [ 4.27549887,  5.37432957,  7.85104799, 13.14586329],
          [ 6.15131307,  8.06423998, 11.61217093, 18.09868288],])

sqrt2 = 2**0.5
bytes_list = [int(sqrt2 ** x) for x in range(40, 56)]
hashes_list = [2**x for x in range(7, 11)]


fig, (ax1, ax2) = plt.subplots(1, 2)

plt.sca(ax1)
plot_speedups(xeon1_grid / ray_pool_grid, bytes_list, hashes_list)

plt.sca(ax2)
plot_speedups(xeon1_grid / ray_tasks_grid, bytes_list, hashes_list)
plt.show()
