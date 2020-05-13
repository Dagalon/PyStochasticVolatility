import numpy as np

from scipy.stats import linregress
from Tools import Types, RNG


def get_mean_ratio_rs(x_t: Types.ndarray, chunksize: int = 1):
    no_elements = len(x_t)
    index = list(range(0, no_elements, chunksize))
    no_packages = len(index)

    mean_n = np.zeros(no_packages - 1)
    r_n = np.zeros(no_packages - 1)
    s_n = np.zeros(no_packages - 1)
    ratio = np.zeros(no_packages - 1)

    for i in range(1, no_packages):
        mean_n[i - 1] = np.mean(x_t[index[i - 1]:index[i]])
        s_n[i - 1] = np.std(x_t[index[i - 1]:index[i]])
        z_n = np.cumsum(x_t[index[i - 1]:index[i]] - mean_n[i - 1])
        r_n[i - 1] = z_n.max() - z_n.min()
        ratio[i - 1] = r_n[i - 1] / s_n[i - 1]

    return np.mean(ratio)


def get_estimator_rs(x_t: Types.ndarray, lower_chunksize: int = 0, upper_chunksize: int = 1):
    # We will suppose that the length of x_t is 2**n where n is some integer > 0.
    # In addition, upper_bound is such that 2**upper_bound < len(x_t)
    no_elements = len(x_t)
    rs = []
    log_no_elements = []
    for i in range(lower_chunksize, upper_chunksize + 1):
        rs.append(get_mean_ratio_rs(x_t, 2 ** i))
        log_no_elements.append(i * np.log(2))

    output = linregress(log_no_elements, np.log(rs))
    return output.slope
