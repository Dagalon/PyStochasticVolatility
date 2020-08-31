import numpy as np

from Tools.Types import Vector, ndarray, HESTON_OUTPUT
from MC_Engines.MC_ExpGaussian1f import ExpGaussianTools, VarianceMC
from Tools import Functionals, Types


def get_path_multi_step(t0: float,
                        t1: float,
                        parameters: Vector,
                        f0: float,
                        v0: float,
                        no_paths: int,
                        no_time_steps: int,
                        type_random_numbers: Types.TYPE_STANDARD_NORMAL_SAMPLING,
                        rnd_generator) -> ndarray:

    mu = parameters[0] # speed reversion
    v = parameters[1] # vol of vol

    no_paths = 2 * no_paths if type_random_numbers == Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

    t_i = np.linspace(t0, t1, no_time_steps)
    delta_t_i = np.diff(t_i)

    f_t = np.empty((no_paths, no_time_steps))
    f_t[:, 0] = f0

    ln_x_t_paths = np.zeros(shape=(no_paths, no_time_steps))
    v_t_paths = np.zeros(shape=(no_paths, no_time_steps))

    ln_x_t_paths[:, 0] = np.log(f0)
    v_t_paths[:, 0] = v0

    map_out_put = {}

    for i in range(1, no_time_steps):
        u_variance = rnd_generator.uniform(0.0, 1.0, no_paths)
        z_f = rnd_generator.normal(0.0, 1.0, no_paths, type_random_numbers)