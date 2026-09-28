"""
Symbolic tools for linearised gravity on spherically symmetric backgrounds.

Everything here is brute-force differential geometry (no shortcuts), so that the
reduced perturbation equations and junction conditions derived with it can be
checked against the literature (Regge & Wheeler 1957; Zerilli 1970;
Martel & Poisson 2005, hereafter MP05) instead of being copied from it.

Angular dependence: axisymmetric (m = 0) modes with a generic function Y(theta)
obeying the Legendre equation  Y'' = -cot(theta) Y' - lam Y,  lam = l(l+1).
Keeping Y generic makes all results valid for arbitrary l.
"""
import sympy as sp

t, r, th, ph = sp.symbols('t r theta phi', real=True)
eps = sp.Symbol('epsilon')
lam = sp.Symbol('lambda', positive=True)          # l(l+1)
Yf = sp.Function('Y')


def legendre_reduce(expr, Y=Yf, x=th, lam=lam):
    """Eliminate Y'' (and higher) with the Legendre equation (MP05 Sec. III)."""
    d1 = sp.Derivative(Y(x), x)
    rule2 = -sp.cot(x) * d1 - lam * Y(x)
    for order in (4, 3, 2):
        # build rule for order-n derivative by differentiating rule2
        rule = rule2
        for _ in range(order - 2):
            rule = sp.diff(rule, x)
            rule = rule.subs(sp.Derivative(Y(x), (x, 2)), rule2)
        expr = expr.subs(sp.Derivative(Y(x), (x, order)), rule)
    return expr


_u = sp.Symbol('u_tanhalf')


def trig_rational(e, x=th):
    """Replace sin, cos, cot, tan of x by rational functions of u = tan(x/2) and cancel.
    If e is secretly x-independent the result contains no u."""
    e = e.subs({sp.cot(x): sp.cos(x) / sp.sin(x), sp.tan(x): sp.sin(x) / sp.cos(x)})
    e = e.subs({sp.sin(x): 2 * _u / (1 + _u**2), sp.cos(x): (1 - _u**2) / (1 + _u**2)})
    return sp.cancel(sp.together(e))


def christoffel(g, ginv, X):
    """Gamma^a_{bc} = 1/2 g^{ad}(d_b g_dc + d_c g_db - d_d g_bc)."""
    n = len(X)
    dg = [[[sp.diff(g[a, b], X[c]) for c in range(n)] for b in range(n)] for a in range(n)]
    G = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(b, n):
                s = 0
                for d in range(n):
                    if ginv[a, d] != 0:
                        s += ginv[a, d] * (dg[d][c][b] + dg[d][b][c] - dg[b][c][d])
                G[a][b][c] = G[a][c][b] = s / 2
    return G


def ricci(G, X):
    """R_{bd} = d_a G^a_{bd} - d_d G^a_{ba} + G^a_{ae} G^e_{bd} - G^a_{de} G^e_{ba}."""
    n = len(X)
    Ric = sp.zeros(n, n)
    for b in range(n):
        for d in range(b, n):
            s = 0
            for a in range(n):
                s += sp.diff(G[a][b][d], X[a]) - sp.diff(G[a][b][a], X[d])
                for e in range(n):
                    s += G[a][a][e] * G[e][b][d] - G[a][d][e] * G[e][b][a]
            Ric[b, d] = Ric[d, b] = s
    return Ric


def first_order(expr):
    """Coefficient of epsilon^1 of an expression analytic in epsilon."""
    return sp.diff(expr, eps).subs(eps, 0)


def delta_christoffel(g0inv, G0, h, X):
    """delta Gamma^a_{bc} = 1/2 g0^{ad}(d_b h_dc + d_c h_db - d_d h_bc - 2 h_de Gamma0^e_bc)."""
    n = len(X)
    dG = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(b, n):
                s = 0
                for d in range(n):
                    if g0inv[a, d] == 0:
                        continue
                    term = sp.diff(h[d, c], X[b]) + sp.diff(h[d, b], X[c]) - sp.diff(h[b, c], X[d])
                    for e in range(n):
                        term -= 2 * h[d, e] * G0[e][b][c]
                    s += g0inv[a, d] * term
                dG[a][b][c] = dG[a][c][b] = s / 2
    return dG


def linearised_ricci(g0, h, X):
    """delta R_{bd} (Palatini-type first-order expansion of tools.ricci)."""
    n = len(X)
    g0inv = sp.simplify(g0.inv())
    G0 = christoffel(g0, g0inv, X)
    dG = delta_christoffel(g0inv, G0, h, X)
    dR = sp.zeros(n, n)
    for b in range(n):
        for d in range(b, n):
            s = 0
            for a in range(n):
                s += sp.diff(dG[a][b][d], X[a]) - sp.diff(dG[a][b][a], X[d])
                for e in range(n):
                    s += (dG[a][a][e] * G0[e][b][d] + G0[a][a][e] * dG[e][b][d]
                          - dG[a][d][e] * G0[e][b][a] - G0[a][d][e] * dG[e][b][a])
            dR[b, d] = dR[d, b] = s
    return dR


def linearised_einstein(g0, h, X):
    """delta G_{ab} for g = g0 + eps h on a Ricci-flat background g0.
    (Background vacuum: delta G_ab = delta R_ab - 1/2 g0_ab g0^{cd} delta R_cd.)"""
    g0inv = g0.inv()
    dRic = linearised_ricci(g0, h, X)
    dR = sum(g0inv[a, b] * dRic[a, b] for a in range(4) for b in range(4))
    return dRic - g0 * dR / 2
