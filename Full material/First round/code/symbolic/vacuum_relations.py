"""
Vacuum perturbation relations in Regge-Wheeler (RW) gauge, frequency domain e^{-i w t}.

Derives (does not assume) the map between the RW-gauge metric functions and the
gauge-invariant master functions of Martel & Poisson (2005) [MP05]:

  odd : Cunningham-Price-Moncrief Psi_odd, MP05 Eq. (5.13)
  even: Zerilli-Moncrief Psi_even,         MP05 Eq. (4.23)-(4.24)

and verifies that the master functions obey the Regge-Wheeler (MP05 Eq. 5.15;
Regge & Wheeler 1957 Eq. 25) and Zerilli (MP05 Eq. 4.26; Zerilli 1970 Eq. 5)
equations, by brute-force linearisation of the Einstein tensor (tools.py).

The results are valid for any M >= 0; M = 0 gives the flat interior.
Output: vacuum_relations.pkl with sympy expressions used by derive_junction.py.
"""
import pickle
import sympy as sp
from tools import t, r, th, ph, lam, Yf, legendre_reduce, linearised_einstein, trig_rational

M = sp.Symbol('M', nonnegative=True)
w = sp.Symbol('omega')
f = 1 - 2 * M / r
E = sp.exp(-sp.I * w * t)
Y = Yf(th)
Yp = sp.diff(Y, th)
X = [t, r, th, ph]
g0 = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(th)**2)
mu = lam - 2                                   # (l-1)(l+2)

Psi = sp.Function('Psi')(r)
V_RW = f * (lam / r**2 - 6 * M / r**3)                                  # MP05 (5.15) x f
n_ = mu / 2                                                             # Zerilli's lambda
V_Z = f * (2 * n_**2 * (n_ + 1) * r**3 + 6 * n_**2 * M * r**2 + 18 * n_ * M**2 * r + 18 * M**3) \
    / (r**3 * (n_ * r + 3 * M)**2)                                     # Zerilli (1970) Eq. (5)
Lam = mu + 6 * M / r                                                    # MP05 (4.24)
V_MP = f / Lam**2 * (mu**2 * ((mu + 2) / r**2 + 6 * M / r**3)
                     + 36 * M**2 / r**4 * (mu + 2 * M / r))            # MP05 (4.26) x f

aY, bY = sp.symbols('aY bY')


def c(e):
    """cheap canonical simplification for rational expressions"""
    return sp.cancel(sp.together(e))


def proj(expr):
    """expr = A(r) Y + B(r) Y'  ->  (A, B); expr is linear in exp(-i w t)."""
    e = sp.expand(expr * sp.exp(sp.I * w * t))
    e = sp.expand(legendre_reduce(e))
    e = e.subs(sp.Derivative(Y, th), bY).subs(Y, aY)
    A, B = trig_rational(e.coeff(aY)), trig_rational(e.coeff(bY))
    return c(A), c(B)


def psi2(V):
    """Psi'' from d^2Psi/dr*^2 + (w^2 - V) Psi = 0 with dr*/dr = 1/f."""
    dP = sp.diff(Psi, r)
    return ((V - w**2) * Psi / f - sp.diff(f, r) * dP) / f


def reduce_psi(expr, V):
    for _ in range(3):
        expr = expr.subs(sp.Derivative(Psi, (r, 3)), sp.diff(psi2(V), r))
        expr = expr.subs(sp.Derivative(Psi, (r, 2)), psi2(V))
    return c(sp.expand(expr))


results = {}
print('Zerilli (1970) Eq.5 vs MP05 Eq.4.26 (times f):', c(V_Z - V_MP), flush=True)

# ---------------------------------------------------------------- ODD PARITY
h0, h1 = sp.Function('h0')(r), sp.Function('h1')(r)
h = sp.zeros(4, 4)
# X_phi = sin(theta) dY/dtheta  (MP05 Eq. 3.2, m=0);  p_aB = h_a X_B  (MP05 Eq. 5.2)
h[0, 3] = h[3, 0] = h0 * E * sp.sin(th) * Yp
h[1, 3] = h[3, 1] = h1 * E * sp.sin(th) * Yp
dG = linearised_einstein(g0, h, X)
h0_psi = f / 2 * sp.diff(r * Psi, r)
h1_psi = -sp.I * w * r * Psi / (2 * f)
odd_res = []
for (a, b) in [(0, 3), (1, 3)]:
    A, B = proj(dG[a, b].subs({h0: h0_psi, h1: h1_psi}).doit() / sp.sin(th))
    odd_res += [reduce_psi(A, V_RW), reduce_psi(B, V_RW)]
e23 = sp.expand(dG[2, 3].subs({h0: h0_psi, h1: h1_psi}).doit() * sp.exp(sp.I * w * t))
e23 = sp.expand(legendre_reduce(e23)).subs(sp.Derivative(Y, th), bY).subs(Y, aY)
odd_res.append(reduce_psi(trig_rational(e23), V_RW))
print('odd: vacuum Einstein residuals with h_t=(f/2)(r Psi)\', h_r=-i w r Psi/(2f):', odd_res, flush=True)
cpm = 2 * r / mu * (sp.diff(h0_psi, r) - 2 * h0_psi / r + sp.I * w * h1_psi)
print('odd: MP05 (5.13) reproduces Psi:', reduce_psi(cpm - Psi, V_RW), flush=True)
results['odd'] = dict(ht=h0_psi, hr=h1_psi)

# ---------------------------------------------------------------- EVEN PARITY
H0, H1, H2, K = [sp.Function(n)(r) for n in ('H0', 'H1', 'H2', 'K')]
h = sp.zeros(4, 4)
h[0, 0] = f * H0 * E * Y
h[0, 1] = h[1, 0] = H1 * E * Y
h[1, 1] = H2 / f * E * Y
h[2, 2] = r**2 * K * E * Y
h[3, 3] = r**2 * sp.sin(th)**2 * K * E * Y
dG = linearised_einstein(g0, h, X)
print('even: dG computed', flush=True)
eqs = {}
for (a, b) in [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2)]:
    A, B = proj(dG[a, b])
    assert A == 0 or B == 0, (a, b)
    eqs[(a, b)] = A if B == 0 else B
trA, trB = proj((dG[2, 2] + dG[3, 3] / sp.sin(th)**2) / 2)
tfA, tfB = proj(dG[2, 2] - dG[3, 3] / sp.sin(th)**2)
print('even: tracefree angular equation (coeff of Y\', Y):', tfB, '|', tfA, flush=True)
Hh = sp.Function('H')(r)
sub_H = {H0: Hh, H2: Hh}
eqs = {k: c(v.subs(sub_H).doit()) for k, v in eqs.items()}
trace_eq = c(trA.subs(sub_H).doit())

dK, dH, dH1 = sp.symbols('dK dH dH1')
rep = {sp.Derivative(K, r): dK, sp.Derivative(Hh, r): dH, sp.Derivative(H1, r): dH1}
sysm = [eqs[k].subs(rep) for k in [(0, 1), (0, 2), (1, 2)]]
sol = sp.solve(sysm, [dK, dH, dH1], dict=True)[0]
sol = {k: c(v) for k, v in sol.items()}
print('even: first-order system  K\' =', sol[dK], flush=True)
print('                          H\' =', sol[dH])
print('                         H1\' =', sol[dH1])
alg = c(eqs[(1, 1)].subs(rep).subs(sol))
print('even: algebraic identity (from G_rr):', alg, flush=True)
Hsol = c(sp.solve(alg, Hh)[0])

Kf, H1f = sp.symbols('Kf H1f')
back = {Kf: K, H1f: H1}
fwd = {K: Kf, H1: H1f}
Kp = sol[dK].subs(Hh, Hsol)
Psi_expr = c((2 * r / lam * (K + 2 * f / Lam * (Hsol - r * Kp))).subs(fwd))


def ddr(expr):
    e = sp.diff(expr.subs(back), r)
    e = e.subs({sp.Derivative(K, r): sol[dK], sp.Derivative(H1, r): sol[dH1]}).subs(Hh, Hsol)
    return c(e.subs(fwd))


dPsi_expr = ddr(Psi_expr)
P, dP = sp.symbols('P dP')
mat = sp.Matrix([[sp.diff(Psi_expr, Kf), sp.diff(Psi_expr, H1f)],
                 [sp.diff(dPsi_expr, Kf), sp.diff(dPsi_expr, H1f)]]).applyfunc(c)
inv = mat.inv().applyfunc(c)
K_psi = c(inv[0, 0] * P + inv[0, 1] * dP)
H1_psi = c(inv[1, 0] * P + inv[1, 1] * dP)
H_psi = c(Hsol.subs(fwd).subs({Kf: K_psi, H1f: H1_psi}))
for nm, e in (('K', K_psi), ('H1', H1_psi), ('H', H_psi)):
    print('even:', nm, '=', sp.collect(sp.expand(e), [P, dP], c), flush=True)

results['even'] = dict(K=K_psi, H1=H1_psi, H=H_psi, P=P, dP=dP)
results['symbols'] = dict(M=M, w=w, r=r, lam=lam)
results['V'] = dict(RW=V_RW, Z=V_MP)
with open('vacuum_relations.pkl', 'wb') as fh:
    pickle.dump(results, fh)
print('saved vacuum_relations.pkl', flush=True)

# ---- verification: every Einstein equation holds once Psi obeys the Zerilli equation.
# Exact check at random rational points (r, M, w, lambda) with Psi, Psi' free symbols:
# after eliminating Psi'' (and Psi''') with the Zerilli equation each residual is
# linear in (Psi, Psi') with rational coefficients, which must vanish identically.
import random
sb = {K: K_psi.subs({P: Psi, dP: sp.diff(Psi, r)}),
      H1: H1_psi.subs({P: Psi, dP: sp.diff(Psi, r)}),
      Hh: H_psi.subs({P: Psi, dP: sp.diff(Psi, r)})}
chk = 2 * r / lam * (sb[K] + 2 * f / Lam * (sb[Hh] - r * sp.diff(sb[K], r)))
allres = [e.subs(sb).doit() for e in list(eqs.values()) + [trace_eq]] + [chk - Psi]
p0, p1 = sp.symbols('p0 p1')
random.seed(1)
worst = 0
for trial in range(4):
    vals = {r: sp.Rational(random.randint(25, 90), 7), M: sp.Rational(random.randint(1, 9), 5),
            w: sp.Rational(random.randint(1, 40), 13), lam: random.choice([6, 12, 20, 30])}
    for e in allres:
        ee = e
        for _ in range(3):
            ee = ee.subs(sp.Derivative(Psi, (r, 3)), sp.diff(psi2(V_MP), r))
            ee = ee.subs(sp.Derivative(Psi, (r, 2)), psi2(V_MP))
        ee = ee.subs(sp.Derivative(Psi, r), p1).subs(Psi, p0).subs(vals)
        ee = sp.nsimplify(sp.expand(ee))
        worst = max(worst, abs(complex(ee.coeff(p0).evalf())), abs(complex(ee.coeff(p1).evalf())))
print('even: max residual of all 6 vacuum Einstein equations + MP05 (4.23) at 4 random'
      ' rational points (exact arithmetic):', worst, flush=True)
