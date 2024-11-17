from sympy import *

y, nu, rho, rho_inv = symbols('y nu rho rho_inv')

d_1_t = nu**3 * rho**3 * ( (y**4 - 2 * y**2 - 1) / 12.0 - (y * y - 1.0) / 3.0)
d_2_t = - nu**2 * rho**2 * (y * y - 1) * y / 6 - 2 * nu**2 * rho_inv**2 * y / 18.0
d_3_t = nu**2 * rho**2 * (4.0 * y * y + 1.0) / 12.0
d = d_1_t + d_2_t + d_3_t

res = d.simplify()
# print(latex(d.simplify(rational=True)))


# T^{3/2}
e_t_1 = nu**3 * rho**3 * ((y**3 + y) / 24.0 - y / 6.0)
e_t_2 =  nu**3 * rho**3 * (y**3 + y) / 8.0 + 0.5 * nu**3 * rho_inv * rho_inv * rho * y

e_t = e_t_1 + e_t_2
res2 = e_t.simplify()
print(latex(e_t.simplify(rational=True)))