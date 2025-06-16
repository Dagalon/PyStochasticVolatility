import sympy as sp
from sympy import print_latex

F, K, T, v = sp.symbols('F, K, T, v')

d = (F - K) / (v * sp.sqrt(T))
pdf = sp.exp(-d * d / 2) / sp.sqrt(2 * sp.pi)

H = - d * pdf / (v * v * T)

HS3 = sp.Derivative(H, F, 3)

print_latex(sp.simplify(HS3.subs(K, F)))
