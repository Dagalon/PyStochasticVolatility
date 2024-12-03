import sympy as sp
from mpmath import fraction

y, alpha, nu, rho, rho_inv = sp.symbols('y, alpha, nu, rho, rho_inv')

# T g_3
b_t = (nu * rho)**2 * (2.0 * y * y - 1.0) / 12.0

# T 0.5 * g^2_2
c_t = (nu * rho * (y * y - 1.0))**2 / 8 + (nu * rho_inv)**2 * (2.0 * y * y + 1.0) / 12

sum_t = b_t + c_t
# print(sp.latex(sum_t.simplify()))

# T^{3/2} g^3_2 / 6
m = ((nu * rho)**3 * (y**4- 6.0 * y**2 - 1.0)/ 8) *  + (nu**3 * (1-rho**2) * rho (2.0 * y**4- y**2 - 1)/4)
partial_m = ((nu * rho)**3  * (y**3 - 3 * y) + nu**3 * rho * (4 * y**3 - y))/2
e_2_t = (y * m  - partial_m) / 6




print(sp.latex(e_2_t.simplify(evaluate=False)))
