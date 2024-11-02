from sympy import *

y, nu, rho, rho_inv = symbols('y nu rho rho_inv')

g2 = (1/8) * (nu * rho * (y * y - 1.0))**2 + (nu * rho_inv)**2 * (2.0 * y * y + 1.0) / 12.0
g1 = (nu * rho)**2 * (y * y - 1.0) / 6.0
g = g1 + g2

res1 = g.simplify(rational=True)

# print(latex(res1))


f1 =  (1/2) * (y * y - 1) * y * (nu * rho)**2 + (nu * rho_inv)**2 * y / 3.0
f2 = y *  ((1/8) * (nu * rho * (y * y - 1.0)**2  + (2.0 * y * y + 1.0) * (nu * rho_inv)**2/ 12.0))
f = f1 + f2

res2 = f.simplify(rational=True)

print(latex(res2))

h1 = (nu * rho)**3 * ((y**3 / 3 - y) / 24 - y / 6)
h2 = (1/4) * nu**3 * rho * y
h3 = (nu * rho)**3 * (y**3 + y) / 8 - (1/2) * (nu * rho)**3 * y
h = h1 + h2 + h3

res3 = h.simplify()
print(latex(h.simplify(rational=True)))