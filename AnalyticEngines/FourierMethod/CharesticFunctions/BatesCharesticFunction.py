
import numba as nb
import numpy as np

@nb.jit("c16[:](f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def get_trap_cf(w, t, x, v, r_t, theta, rho, k, epsilon,jumpmean, jumpstd,jumpintensity, b, u):
    a = k * theta

    p = u * 1j * w - 0.5 * w * w
    q = b - rho * epsilon * 1j * w
    r = 0.5 * epsilon * epsilon

    d = np.sqrt(q * q - 4.0 * p * r)
    aux_b_t = b - rho * epsilon * 1j * w + d
    c = (aux_b_t - 2.0 * d) / aux_b_t

    d_t = ((1.0 - np.exp(-d * t)) / (1.0 - c * np.exp(-d * t))) * (aux_b_t - 2.0 * d) / (epsilon * epsilon)

    aux_c_t = (1.0 - c * np.exp(-d * t)) / (1.0 - c)
    c_t = r_t * 1j * t * w + (a / (epsilon * epsilon)) * ((aux_b_t - 2.0 * d) * t - 2.0 * np.log(aux_c_t))

    jumpvar=0.5*jumpstd*jumpstd
    aux_jump1=np.exp(jumpmean+jumpvar)-1
    aux_jump2=np.exp(jumpmean*1j*w-jumpvar*w*w)-1
    jumpterm=jumpintensity*t*(aux_jump2-1j*w *aux_jump1)

    return np.exp(jumpterm+c_t + d_t * v + 1j * w * x)