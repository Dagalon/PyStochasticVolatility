import numba as nb
import numpy as np


@nb.jit("(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def update_diagonal(diffusion,
                    convection,
                    source,
                    diagonal_gradient,
                    diagonal_laplacian,
                    diagonal_update):
    no_nodes = len(diffusion)
    for i in range(1, no_nodes - 1):
        diagonal_update[i] = diffusion[i] * diagonal_laplacian[i] + \
                             convection[i] * diagonal_gradient[i] + \
                             source[i]


@nb.jit("(f8[:], f8[:], f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def update_diagonal_lower(diffusion,
                          convection,
                          diagonal_lower_gradient,
                          diagonal_lower_laplacian,
                          diagonal_lower_update):
    no_nodes = len(diffusion)
    for i in range(1, no_nodes - 1):
        diagonal_lower_update[i - 1] = diffusion[i] * diagonal_lower_laplacian[i - 1] + \
                                       convection[i] * diagonal_lower_gradient[i - 1]


@nb.jit("(f8[:], f8[:], f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def update_diagonal_upper(diffusion,
                          convection,
                          diagonal_upper_gradient,
                          diagonal_upper_laplacian,
                          diagonal_upper_update):
    no_nodes = len(diffusion)
    for i in range(1, no_nodes - 1):
        diagonal_upper_update[i] = diffusion[i] * diagonal_upper_laplacian[i] + \
                                   convection[i] * diagonal_upper_gradient[i]


@nb.jit("(f8[:], f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def tdr_system_solver(diagonal,
                      diagonal_lower,
                      diagonal_upper,
                      b):
    no_nodes = len(diagonal)
    x = np.zeros(no_nodes)
    gamma = np.zeros(no_nodes)
    rho = np.zeros(no_nodes)
    gamma[0] = diagonal_upper[0] / diagonal[0]
    rho[0] = b[0] / diagonal[0]

    for i in range(1, no_nodes - 1):
        gamma[i] = diagonal_upper[i] / (diagonal[i] - diagonal_lower[i - 1] * gamma[i - 1])
        rho[i] = (b[i] - diagonal_lower[i - 1] * rho[i - 1]) / (diagonal[i] - diagonal_lower[i - 1] * gamma[i - 1])

    x[no_nodes - 1] = (b[no_nodes - 1] - diagonal_lower[no_nodes - 2] * rho[no_nodes - 2]) / \
                      (diagonal[no_nodes - 1] - diagonal_lower[no_nodes - 2] * gamma[no_nodes - 2])

    for i in range(no_nodes - 2, -1, -1):
        x[i] = rho[i] - gamma[i] * x[i + 1]

    return x


@nb.jit("f8[:](f8[:], f8[:], f8[:], f8[:])", nopython=True, nogil=True)
def apply_tdr(diagonal,
              diagonal_lower,
              diagonal_upper,
              b):
    no_nodes = len(diagonal)
    y = np.zeros(no_nodes)
    y[0] = diagonal[0] * b[0] + diagonal_upper[0] * b[1]
    y[no_nodes - 1] = diagonal_lower[no_nodes - 2] * b[no_nodes - 2] + diagonal[no_nodes - 1] * b[no_nodes - 1]
    for i in range(0, no_nodes - 1):
        y[i] = diagonal_lower[i - 1] * b[i - 1] + diagonal[i] * b[i] + diagonal_upper[i] * b[i + 1]

    return y
