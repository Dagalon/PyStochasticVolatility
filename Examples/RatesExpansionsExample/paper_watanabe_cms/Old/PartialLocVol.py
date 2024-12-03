import sympy
from sympy import *

K, f, alpha, nu, rho = symbols('K, f, alpha, nu, rho')

# alpha * np.sqrt(1.0 + 2.0 * rho * nu * y_i + nu * nu * y_i * y_i)

f0 = 0.03

y = (K - f0) / alpha

loc_vol = alpha * sympy.sqrt(1 + 2 * rho * nu * y + nu**2 * y ** 2)

# first partial
partial_K = sympy.diff(loc_vol, K)
partial_K_f0 = partial_K.subs(K, f0)
print(partial_K_f0)

# second partial
partial_KK = sympy.diff(loc_vol, K, K)
partial_KK_f0 = partial_KK.subs(K, f0)
print(partial_KK_f0)

# third partial
partial_KKK = sympy.diff(loc_vol, K, K, K)
partial_KKK_f0 = partial_KKK.subs(K, f0)
print(partial_KKK_f0)