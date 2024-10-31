import sympy
from sympy import symbols, sympify

f, k, t, beta, sigma = symbols('f k t beta sigma')

m = sigma * (f - k) * (1 - beta) / (f**(1 - beta) - k ** (1 - beta))
normal_hagan_sigma = sigma * (f - k) * (1 - beta) / (f**(1 - beta) - k ** (1 - beta)) * (1.0 + t * beta * (beta - 2.0) * sigma * sigma / (24.0 * (0.5 * (f + k)) ** (2.0 * (1 - beta))))

res = sympy.diff(m, k, k)
res_atm = res.limit(k, f)
print(sympy.latex(sympy.sympify(res_atm)))

val = res_atm.subs(k, 0.03).subs(beta, 0.6).subs(sigma, 0.3).subs(f, 0.03)








