import numpy as np

from scipy.stats import linregress
from Tools import Types
from scipy.optimize import curve_fit


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
    rs = []
    log_no_elements = []

    for i in range(lower_chunksize, upper_chunksize + 1):
        rs.append(get_mean_ratio_rs(x_t, 2 ** i))
        log_no_elements.append(i * np.log(2))

    def func(x, a, b):
        return a + b * x

    popt, pcov = curve_fit(func, log_no_elements, np.log(rs))
    estimated_rs = func(np.array(log_no_elements), *popt)

    return popt[0], popt[1], log_no_elements, np.log(rs), estimated_rs
