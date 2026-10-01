"""
Gauge check: repeat the l = 0 and l = 1 mode conditions fixing the shell reparametrization on the EXTERIOR side
(a_p = b_p = 0) instead of the interior side (a_m = b_m = 0). The frequencies must not change.
"""
import sympy as sp
import israel_low_l as I

x, Om, G = I.x, I.Om, I.G
for l in (0, 1):
    amps = sp.symbols('a_m X_m b_m a_p X_p b_p s_0 w_0')
    a_m, X_m, b_m, a_p, X_p, b_p, s0, w0 = amps
    cols = [a_m, X_m, b_m, X_p, s0, w0] if l == 1 else [a_m, X_m, X_p, s0]
    rows = []
    comps = I.junction_rows(l, I.ANGLES[0], amps)
    for k, (z0, v) in comps.items():
        if 'phph' in k:
            continue
        v = v.subs({a_p: 0, b_p: 0})
        if l == 0:
            v = v.subs({b_m: 0, w0: 0})
        row = [sp.cancel(v.coeff(c)) for c in cols]
        if any(r != 0 for r in row):
            rows.append(row)
    A = sp.Matrix(rows)
    det = sp.factor(sp.simplify(A.det()))
    fac = [fa for fa, _ in sp.factor_list(sp.numer(sp.together(det)), Om)[1] if fa.has(Om) and fa != Om]
    w2 = sp.symbols('w2')
    roots = [sp.factor(sp.simplify(r)) for fa in fac for r in sp.solve(fa.subs(Om, sp.sqrt(w2*(1 - x**2)/2)/x), w2)]
    print('l=%d, exterior gauge fixing: omega^2 R^3/M =' % l, roots)
