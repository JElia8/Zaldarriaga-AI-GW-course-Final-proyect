"""
Linearized Israel junction conditions for the l = 0 and l = 1 (even) oscillations of the static fluid shell,
derived from first principles with the METRIC UNPERTURBED on both sides.

Why this is allowed
-------------------
* Interior: flat vacuum. Exterior: Schwarzschild vacuum.
* l = 1, even parity: every vacuum perturbation is pure gauge, in flat space and in Schwarzschild
  (Martel & Poisson 2005, Secs. IV D and V E). l = 0: the only non-gauge vacuum perturbation is a
  change of mass dM (Birkhoff), which is static and so absent from any mode with Omega != 0.
* Hence on each side we may use coordinates in which the metric is exactly the background one. All the
  physics is in the perturbed EMBEDDING of the shell in each side and in the perturbed fluid. The gauge
  transformations of the two sides are independent, and they are absorbed into the two embeddings.

Embedding (y^a = (tau, theta, phi), axisymmetric m = 0, Y = P_l(cos theta)):
    t   = tau/sqrt(F(R)) + eps A Y ,   r = R + eps X Y ,   th' = theta + eps B dY/dtheta ,   ph' = phi
with F = 1 (interior; t is Minkowski time) or F = f = 1 - 2M/r (exterior), and (A, X, B) ~ e^{-i Omega tau}.
Fluid:  S_ab = (sigma + p) u_a u_b + p h_ab ,  sigma -> sigma + eps s Y ,  p -> p + eps v_s^2 s Y ,
        u^a = (1 + eps h1_tautau/2, eps w dY/dtheta, 0)     (normalized: h_ab u^a u^b = -1 + O(eps^2)),
        v_s^2 = Gamma p/(sigma + p)   (the EOS used for the static shell, PSP26 convention).
Equations (Israel 1966; Poisson, Toolkit Secs. 3.6-3.7):
    h^+_ab = h^-_ab ,     [K_ab] - h_ab [K] = -8 pi S_ab ,
    K_ab = -n_mu (d_b e^mu_a + Gamma^mu_{nu rho} e^nu_a e^rho_b) ,  n outward.
Gauge on the shell: intrinsic reparametrizations y^a -> y^a + eps zeta^a shift A_-, A_+ and B_-, B_+
together; we fix A_- = B_- = 0.
Units R = 1, x := sqrt(f(R)), M = (1 - x^2)/2. Omega is the frequency in shell proper time (= interior
Minkowski time); the exterior-time frequency is omega = x Omega.

Every component is evaluated at two generic polar angles with rational sin and cos
(cos theta = 3/5 and 5/13), which keeps all expressions rational. The rows are the 8 components x 2 angles.
Output: ../results/israel_l{0,1}_matrix.pkl (matrix, unknowns, row names) and a log.
"""
import pickle
import sympy as sp

tau, th, ph, eps = sp.symbols('tau theta phi epsilon')
x = sp.symbols('x', positive=True)
Om = sp.symbols('Omega')
G = sp.symbols('Gamma', positive=True)
PI = sp.pi
Rv = sp.Integer(1)
Mv = (1 - x**2)/2
ANGLES = (sp.acos(sp.Rational(3, 5)), sp.acos(sp.Rational(5, 13)))


def lin(e):
    """(zeroth, first) order coefficients in eps."""
    return (sp.cancel(e.subs(eps, 0)), sp.cancel(sp.diff(e, eps).subs(eps, 0)))


def pmul(a, b):
    return (a[0]*b[0], a[0]*b[1] + a[1]*b[0])


def padd(*ps):
    return (sum(p[0] for p in ps), sum(p[1] for p in ps))


def psc(c, a):
    return (c*a[0], c*a[1])


def side(F, amps, Y, th0):
    """Induced metric and extrinsic curvature, as (zeroth, first)-order pairs, at polar angle th0."""
    A, X, B = [a*sp.exp(-sp.I*Om*tau) for a in amps]
    t_, r_, T_, P_ = sp.symbols('t_ r_ T_ P_')
    X4 = [t_, r_, T_, P_]
    g = sp.diag(-F(r_), 1/F(r_), r_**2, r_**2*sp.sin(T_)**2)
    gi = g.inv()
    Gam = [[[sp.cancel(sum(gi[m, k]*(sp.diff(g[k, a], X4[b]) + sp.diff(g[k, b], X4[a]) - sp.diff(g[a, b], X4[k]))
                           for k in range(4))/2) for b in range(4)] for a in range(4)] for m in range(4)]
    xR = sp.sqrt(F(Rv))
    Yt = sp.diff(Y, th)
    emb = [tau/xR + eps*A*Y, Rv + eps*X*Y, th + eps*B*Yt, ph]
    ys = [tau, th, ph]
    e = [[sp.diff(emb[m], ys[a]) for m in range(4)] for a in range(3)]
    de = [[[sp.diff(e[a][m], ys[b]) for m in range(4)] for b in range(3)] for a in range(3)]
    at = {tau: 0, th: th0, ph: 0}
    pos = {X4[m]: emb[m].subs(at) for m in range(4)}
    eP = [[lin(e[a][m].subs(at)) for m in range(4)] for a in range(3)]
    deP = [[[lin(de[a][b][m].subs(at)) for m in range(4)] for b in range(3)] for a in range(3)]
    gP = [[lin(g[m, k].subs(pos)) for k in range(4)] for m in range(4)]
    giP = [[lin(gi[m, k].subs(pos)) for k in range(4)] for m in range(4)]
    GamP = [[[lin(Gam[m][a][b].subs(pos)) for b in range(4)] for a in range(4)] for m in range(4)]
    # normal covector by the cofactor construction n_mu = eps_{mu nu rho sig} e_tau^nu e_th^rho e_ph^sig
    ee = [[e[a][m].subs(at) for m in range(4)] for a in range(3)]
    ncov = []
    for mu in range(4):
        Mm = sp.Matrix([[1 if k == mu else 0 for k in range(4)]] + ee)
        ncov.append(lin(Mm.det()))
    N2 = padd(*[pmul(giP[a][b], pmul(ncov[a], ncov[b])) for a in range(4) for b in range(4)])
    N0 = sp.sqrt(sp.factor(N2[0]))
    sgn = 1 if (ncov[1][0]/N0).subs(x, sp.Rational(7, 10)) > 0 else -1
    inv_norm = (sgn/N0, -sgn*N2[1]/(2*N0**3))
    n = [pmul(inv_norm, ncov[m]) for m in range(4)]
    h = [[None]*3 for _ in range(3)]
    K = [[None]*3 for _ in range(3)]
    for a in range(3):
        for b in range(3):
            h[a][b] = padd(*[pmul(gP[m][k], pmul(eP[a][m], eP[b][k])) for m in range(4) for k in range(4)])
            acc = [padd(deP[a][b][m], *[pmul(GamP[m][p][q], pmul(eP[a][p], eP[b][q]))
                                         for p in range(4) for q in range(4)]) for m in range(4)]
            K[a][b] = psc(-1, padd(*[pmul(n[m], acc[m]) for m in range(4)]))
    h = [[(sp.cancel(v[0]), sp.cancel(v[1])) for v in row] for row in h]
    K = [[(sp.cancel(v[0]), sp.cancel(v[1])) for v in row] for row in K]
    return h, K


def junction_rows(l, th0, amps):
    a_m, X_m, b_m, a_p, X_p, b_p, s0, w0 = amps
    Y = sp.legendre(l, sp.cos(th))
    hm, Km = side(lambda r: sp.Integer(1), (a_m, X_m, b_m), Y, th0)
    hp, Kp = side(lambda r: 1 - 2*Mv/r, (a_p, X_p, b_p), Y, th0)
    Y0 = Y.subs(th, th0)
    Yt0 = sp.diff(Y, th).subs(th, th0)
    sq = x
    sig0 = (1 - sq)/(4*PI*Rv)
    p0 = (1 - sq)/(4*sq)*sig0
    vs2 = G*p0/(sig0 + p0)
    sig = (sig0, s0*Y0)
    pr = (p0, vs2*s0*Y0)
    uup = [(sp.Integer(1), hm[0][0][1]/2), (0, w0*Yt0), (0, 0)]
    ulo = [padd(*[pmul(hm[a][b], uup[b]) for b in range(3)]) for a in range(3)]
    S = [[padd(pmul(padd(sig, pr), pmul(ulo[a], ulo[b])), pmul(pr, hm[a][b])) for b in range(3)] for a in range(3)]
    H0 = sp.Matrix(3, 3, lambda a, b: hm[a][b][0])
    H1 = sp.Matrix(3, 3, lambda a, b: hm[a][b][1])
    Hi0 = H0.inv()
    Hi1 = -Hi0*H1*Hi0
    hinv = [[(Hi0[a, b], Hi1[a, b]) for b in range(3)] for a in range(3)]
    trK = lambda KK: padd(*[pmul(hinv[a][b], KK[a][b]) for a in range(3) for b in range(3)])
    dtr = padd(trK(Kp), psc(-1, trK(Km)))
    comps = {}
    for (a, b, nm) in ((0, 0, 'tt'), (0, 1, 'tth'), (1, 1, 'thth'), (2, 2, 'phph')):
        comps['cont_' + nm] = padd(hp[a][b], psc(-1, hm[a][b]))
        comps['isr_' + nm] = padd(Kp[a][b], psc(-1, Km[a][b]), psc(-1, pmul(hm[a][b], dtr)), psc(8*PI, S[a][b]))
    return {k: (sp.simplify(v[0]), sp.expand(sp.cancel(v[1]))) for k, v in comps.items()}


def mode_matrix(l):
    amps = sp.symbols('a_m X_m b_m a_p X_p b_p s_0 w_0')
    a_m, X_m, b_m, a_p, X_p, b_p, s0, w0 = amps
    cols = [X_m, a_p, X_p, b_p, s0, w0] if l >= 1 else [X_m, a_p, X_p, s0]
    rows, names, zeroth = [], [], {}
    for i, th0 in enumerate(ANGLES):
        comps = junction_rows(l, th0, amps)
        for k, (z0, v) in comps.items():
            zeroth['%s@%d' % (k, i)] = z0
            v = v.subs({a_m: 0, b_m: 0})
            if l == 0:
                v = v.subs({b_p: 0, w0: 0})
            row = [sp.factor(sp.cancel(v.coeff(c))) for c in cols]
            assert sp.cancel(v - sum(r*c for r, c in zip(row, cols))) == 0, k
            if any(r != 0 for r in row):
                rows.append(row)
                names.append('%s@%d' % (k, i))
    return sp.Matrix(rows), cols, names, zeroth


if __name__ == '__main__':
    for l in (0, 1):
        print('=' * 80)
        print('l =', l)
        A, cols, names, zeroth = mode_matrix(l)
        bad = {k: v for k, v in zeroth.items() if v != 0}
        print('zeroth-order residuals (background Israel eqs; must be empty):', bad)
        print('unknowns:', cols, '   number of rows:', len(names))
        for nm, i in zip(names, range(A.rows)):
            print('  %-12s' % nm, list(A.row(i)))
        num = {x: sp.Rational(7, 10), G: 2}
        print('rank at Omega = 3/10:', A.subs(num).subs(Om, sp.Rational(3, 10)).rank(),
              '   at Omega = 0:', A.subs(num).subs(Om, 0).rank())
        pickle.dump((A, cols, names), open('../results/israel_l%d_matrix.pkl' % l, 'wb'))
