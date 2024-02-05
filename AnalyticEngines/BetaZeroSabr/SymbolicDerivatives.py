import sympy as sym


def lv_normal_sabr(f, f0, alpha, rho, nu):
    return alpha * sym.sqrt(1 + nu * nu * (f - f0) / alpha * (f - f0) / alpha + 2 * rho * nu * (f - f0) / alpha)


x, forward, a, r, v = sym.symbols('x forward a r v')

eq_sabr = lv_normal_sabr(x, forward, a, r, v)


# partial_f
partial_f = sym.simplify(eq_sabr.diff(x))
res1 = partial_f.evalf(subs={x: 0.03, forward: 0.03})
print(res1)

# partial_ff
partial_ff = eq_sabr.diff(x, x)
res2 = partial_ff.evalf(subs={x: 0.03, forward: 0.03})
print(res2)

# partial_f (partial_f * f)
eq_prod = partial_f * eq_sabr
res3 = sym.simplify(eq_prod.diff(x).evalf(subs={x: 0.03, forward: 0.03}))
print(res3)

# partial_ff (partial_f * f)
eq_prod = partial_f * eq_sabr
res4 = sym.simplify(eq_prod.diff(x, x).evalf(subs={x: 0.03, forward: 0.03}))
print(res4)

# partial_f ((partial_f * f) * f^2)
eq_prod = eq_prod.diff(x, x) * eq_sabr * eq_sabr
res5 = sym.simplify(eq_prod.diff(x).evalf(subs={x: 0.03, forward: 0.03}))
print(res5)

