"""
First-principles derivation of the linearised Israel junction conditions for a
spherical thin shell (flat interior / Schwarzschild exterior), both parities.

Method (see derivation document, Sec. 4):
  * each side is described in its own Regge-Wheeler (RW) gauge;
  * the shell world-tube is embedded in each side with its own perturbed
    embedding x_s^mu(y) = X0_s^mu(y) + eps Z_s^mu(y),  y = (tau, theta, phi);
    intrinsic coordinates are chosen to coincide with the exterior ones, so
    Z_+ = (0, xi_+ Y, 0, 0) and Z_- = (zeta Y, xi_- Y, eta Y', 0) [+ odd twist];
  * induced metric h_ab = g_{mu nu} e^mu_a e^nu_b and extrinsic curvature
    K_ab = -n_mu (d_a e^mu_b + Gamma^mu_{nu rho} e^nu_a e^rho_b)
    (Poisson 2004, Sec. 3.1 & 3.7; Ipser & Sikivie 1984 Eqs. 2.1-2.3) are
    computed to first order in eps by brute force;
  * junction conditions: [h_ab] = 0 (Poisson Eq. 3.49) and
    [K_ab - h_ab K] = -8 pi S_ab (Ipser & Sikivie Eq. 2.6; Poisson Eq. 3.56);
  * bulk metric functions are replaced by the master functions using the
    vacuum relations derived and verified in vacuum_relations.py.

Shell models
  C : static perfect-fluid shell, S_ab = (sig+p) u_a u_b + p h_ab, background
      sig, p from Poisson (2004) Eqs. (3.80)-(3.81); perturbation EOS
      delta p = vs2 * delta sig  (cf. Pani et al. 2009 Eq. 3.33).
  A : domain wall S_ab = -sig h_ab (Ipser & Sikivie Eq. 2.10), frozen at the
      turning point Rdot = 0 with Rddot from Ipser & Sikivie Eq. (3.7a).

Output: junction_results.pkl with 2x2 transfer matrices
        (Psi_+, dPsi_+/dr) = T (Psi_-, dPsi_-/dr)   at r = R.
"""
import pickle
import sympy as sp
from tools import lam, Yf, legendre_reduce, christoffel, delta_christoffel, trig_rational, _u

tau, th, ph = sp.symbols('tau theta phi', real=True)
t, r = sp.symbols('t r', real=True)
eps = sp.Symbol('epsilon')
M = sp.Symbol('M', positive=True)
w = sp.Symbol('omega')          # exterior (Schwarzschild-time) frequency
Rs = sp.Symbol('R', positive=True)
Rdd = sp.Symbol('Rdd')          # Rddot at the frozen instant (model A)
Y = Yf(th)
Yp = sp.diff(Y, th)
mu = lam - 2

with open('vacuum_relations.pkl', 'rb') as fh:
    VAC = pickle.load(fh)
Mv, wv, rv = VAC['symbols']['M'], VAC['symbols']['w'], VAC['symbols']['r']

def cz(e):
    """cheap canonical form: trig of theta -> rational in tan(theta/2), then cancel"""
    return trig_rational(e)


Rf = sp.Function('Rt')(tau)                   # shell radius R(tau)
T0 = sp.Function('T0')(tau)                   # background t(tau) on each side


def side_geometry(F, parity, zpert):
    """First-order induced metric and extrinsic curvature of the shell as seen
    from one side with metric -F dt^2 + dr^2/F + r^2 dOmega^2 plus an RW-gauge
    perturbation (bulk functions of r times exp(-i w_s t)).

    zpert: dict of embedding perturbation functions of tau (xi, zeta, eta, chi).
    Returns h0, h1, K0, K1 (3x3 sympy matrices in y = (tau, theta, phi)), each
    still containing the bulk symbols listed in 'bulk'.
    """
    X = [t, r, th, ph]
    g0 = sp.diag(-F, 1 / F, r**2, r**2 * sp.sin(th)**2)
    g0inv = sp.diag(-1 / F, F, 1 / r**2, 1 / (r**2 * sp.sin(th)**2))
    ws = sp.Symbol('w_s')
    E = sp.exp(-sp.I * ws * t)
    h = sp.zeros(4, 4)
    bulk = {}
    if parity == 'even':
        Hh, H1, K = [sp.Function(n)(r) for n in ('H', 'H1', 'K')]
        h[0, 0] = F * Hh * E * Y
        h[0, 1] = h[1, 0] = H1 * E * Y
        h[1, 1] = Hh / F * E * Y
        h[2, 2] = r**2 * K * E * Y
        h[3, 3] = r**2 * sp.sin(th)**2 * K * E * Y
        funcs = [Hh, H1, K]
    else:
        h0, h1 = [sp.Function(n)(r) for n in ('h0', 'h1')]
        h[0, 3] = h[3, 0] = h0 * E * sp.sin(th) * Yp
        h[1, 3] = h[3, 1] = h1 * E * sp.sin(th) * Yp
        funcs = [h0, h1]
    G0 = christoffel(g0, g0inv, X)
    dG = delta_christoffel(g0inv, G0, h, X)
    dginv = -g0inv * h * g0inv

    # embedding
    xi = zpert.get('xi', 0)
    zeta = zpert.get('zeta', 0)
    eta = zpert.get('eta', 0)
    chi = zpert.get('chi', 0)
    X0 = [T0, Rf, th, ph]
    Z = [zeta * Y, xi * Y, eta * Yp, chi * Yp / sp.sin(th)]
    Yc = [tau, th, ph]
    e0 = [[sp.diff(X0[m], Yc[a]) for m in range(4)] for a in range(3)]
    e1 = [[sp.diff(Z[m], Yc[a]) for m in range(4)] for a in range(3)]
    at0 = {t: T0, r: Rf}

    def on_shell(expr):
        """value at X0(y), bulk functions -> symbols"""
        e = expr
        for fn in funcs:
            nm = str(fn.func)
            e = e.subs(sp.Derivative(fn, (r, 2)), sp.Symbol(nm + '_rr'))
            e = e.subs(sp.Derivative(fn, r), sp.Symbol(nm + '_r'))
            e = e.subs(fn, sp.Symbol(nm))
        return e.subs(at0)

    def pert_at_shell(Q0, dQ):
        """first-order part of Q(x(y)) = Q0(X0+eps Z) + eps dQ(X0)"""
        s = on_shell(dQ)
        for m in range(4):
            if Z[m] != 0:
                s += Z[m] * on_shell(sp.diff(Q0, X[m]))
        return s

    # metric and inverse on shell
    g_0 = [[on_shell(g0[m, n]) for n in range(4)] for m in range(4)]
    g_1 = [[pert_at_shell(g0[m, n], h[m, n]) for n in range(4)] for m in range(4)]
    gi_0 = [[on_shell(g0inv[m, n]) for n in range(4)] for m in range(4)]
    gi_1 = [[pert_at_shell(g0inv[m, n], dginv[m, n]) for n in range(4)] for m in range(4)]
    Ga_0 = [[[on_shell(G0[a][b][c]) for c in range(4)] for b in range(4)] for a in range(4)]
    Ga_1 = [[[pert_at_shell(G0[a][b][c], dG[a][b][c]) for c in range(4)] for b in range(4)]
            for a in range(4)]

    # induced metric
    hind0 = sp.zeros(3, 3)
    hind1 = sp.zeros(3, 3)
    for a in range(3):
        for b in range(3):
            s0 = s1 = 0
            for m in range(4):
                for n in range(4):
                    s0 += g_0[m][n] * e0[a][m] * e0[b][n]
                    s1 += (g_1[m][n] * e0[a][m] * e0[b][n] + g_0[m][n] *
                           (e1[a][m] * e0[b][n] + e0[a][m] * e1[b][n]))
            hind0[a, b], hind1[a, b] = s0, s1

    # unit normal n_mu = n0 + eps n1, outward (increasing r)
    Rd = sp.diff(Rf, tau)
    Td = sp.diff(T0, tau)
    nrm0 = [-Rd, Td, 0, 0]
    # normalise n0 (exact for the background)
    N0 = 1 / sp.sqrt(sum(gi_0[m][n] * nrm0[m] * nrm0[n] for m in range(4) for n in range(4)))
    n0 = [N0 * c for c in nrm0]
    a_ = sp.symbols('n1_0:4')
    eqs = []
    for a in range(3):
        eqs.append(sum(a_[m] * e0[a][m] + n0[m] * e1[a][m] for m in range(4)))
    eqs.append(sum(2 * gi_0[m][n] * n0[m] * a_[n] + gi_1[m][n] * n0[m] * n0[n]
                   for m in range(4) for n in range(4)))
    sol = sp.solve(eqs, a_, dict=True)[0]
    n1 = [sol[a_[m]] for m in range(4)]

    # extrinsic curvature K_ab = -n_mu (d_a e^mu_b + Gamma^mu_{nu rho} e^nu_a e^rho_b)
    K0 = sp.zeros(3, 3)
    K1 = sp.zeros(3, 3)
    for a in range(3):
        for b in range(3):
            acc0, acc1 = [], []
            for m in range(4):
                A0 = sp.diff(e0[b][m], Yc[a])
                A1 = sp.diff(e1[b][m], Yc[a])
                for nu in range(4):
                    for rho in range(4):
                        A0 += Ga_0[m][nu][rho] * e0[a][nu] * e0[b][rho]
                        A1 += (Ga_1[m][nu][rho] * e0[a][nu] * e0[b][rho] + Ga_0[m][nu][rho] *
                               (e1[a][nu] * e0[b][rho] + e0[a][nu] * e1[b][rho]))
                acc0.append(-n0[m] * A0)
                acc1.append(-n1[m] * A0 - n0[m] * A1)
            K0[a, b] = sum(acc0)
            K1[a, b] = sum(acc1)
    return dict(h0=hind0, h1=hind1, K0=K0, K1=K1, ws=ws,
                bulk=[str(fn.func) for fn in funcs])


def background_subs(F, dynamic):
    """Rules for R(tau), T0(tau) and their derivatives at the (frozen) instant."""
    Rdot = sp.Symbol('Rdot')
    FR = F.subs(r, Rs)
    beta = sp.sqrt(FR + Rdot**2)
    Td = beta / FR                    # Ipser & Sikivie Eq. (3.3)-(3.4); = alpha for F = 1
    # d/dtau of Td along the motion: dTd/dR * Rdot + dTd/dRdot * Rddot
    Tdd = sp.diff(Td, Rs) * Rdot + sp.diff(Td, Rdot) * (Rdd if dynamic else 0)
    rules = [(sp.Derivative(T0, (tau, 2)), Tdd), (sp.Derivative(T0, tau), Td),
             (sp.Derivative(Rf, (tau, 2)), Rdd if dynamic else 0),
             (sp.Derivative(Rf, tau), Rdot), (Rf, Rs)]
    return rules, Rdot


def freeze(expr, F, dynamic, nu):
    """Evaluate at the instant: Rdot = 0, harmonic time dependence e^{-i nu tau}."""
    rules, Rdot = background_subs(F, dynamic)
    e = expr
    for a, b in rules:
        e = e.subs(a, b)
    e = e.subs(Rdot, 0)
    e = e.subs(sp.exp(-sp.I * sp.Symbol('w_s') * T0), 1)
    return e


def time_harmonic(expr, funcs, nu):
    """Replace embedding perturbation functions of tau by amplitudes x e^{-i nu tau}
    (the common factor is dropped)."""
    e = expr
    for fn in funcs:
        s = sp.Symbol(str(fn.func) + '_a')
        e = e.subs(sp.Derivative(fn, (tau, 2)), -nu**2 * s)
        e = e.subs(sp.Derivative(fn, tau), -sp.I * nu * s)
        e = e.subs(fn, s)
    return e


def _numcheck(e, extra=None, tries=2):
    """max |e| at random complex values of all free symbols (numerical zero test)."""
    import random
    worst = 0.0
    for k in range(tries):
        rnd = random.Random(1234 + k)
        vals = {s: sp.Rational(rnd.randint(300, 1700), 1000) + sp.I * sp.Rational(rnd.randint(-500, 500), 1000)
                for s in e.free_symbols}
        if extra:
            vals.update(extra)
        worst = max(worst, abs(complex(e.subs(vals).evalf(40))))
    return worst


def _thetafree(c, name):
    """Check numerically that coefficient c does not depend on theta; return c(theta=pi/2)."""
    import random
    if not c.has(th):
        return c
    rnd = random.Random(99)
    vals = {s: sp.Rational(rnd.randint(300, 1700), 1000) + sp.I * sp.Rational(rnd.randint(-500, 500), 1000)
            for s in c.free_symbols if s != th}
    c1 = complex(c.subs(vals).subs(th, sp.Rational(7, 10)).evalf(40))
    c2 = complex(c.subs(vals).subs(th, sp.Rational(13, 10)).evalf(40))
    assert abs(c1 - c2) <= 1e-25 * max(1.0, abs(c1)), (name, c1, c2)
    return c.subs(th, sp.pi / 2)


def harmonic_components(T, parity):
    """Project a first-order 3-tensor on Sigma onto the l-harmonics (MP05 Sec. III).

    Y, Y', Y'' are treated as independent symbols (no Legendre reduction):
      even:  T_tt = A_tt Y,  T_tA = A_tA D_A Y,
             T_AB = A_tr Omega_AB Y + A_tf Y_AB,  Y_AB = D_A D_B Y + lam/2 Omega_AB Y
             -> A_tf = coeff of Y'' in T_thth,  A_tr = coeff of Y in T_thth - lam/2 A_tf,
             and T_phph / sin^2 must equal A_tr Y + A_tf (cot Y' + lam/2 Y)  [checked].
      odd:   T_tphi = A_tA sin(th) Y',  T_thphi = A_AB X_thphi,
             X_thphi = 1/2 sin th (Y'' - cot th Y').
    All structural claims (vanishing components, theta independence) are verified
    numerically at random parameter values; coefficients are then evaluated at theta = pi/2."""
    Y0, Y1, Y2, Y3 = sp.symbols('Y0 Y1 Y2 Y3')

    def sym(e):
        e = e.subs(sp.Derivative(Y, (th, 3)), Y3).subs(sp.Derivative(Y, (th, 2)), Y2)
        e = e.subs(sp.Derivative(Y, th), Y1).subs(Y, Y0)
        return sp.expand(e)

    def co(e, s):
        return e.coeff(s)

    out = {}
    if parity == 'even':
        tt, tA, thth, phph = sym(T[0, 0]), sym(T[0, 1]), sym(T[1, 1]), sym(T[2, 2])
        for e, bad, nm in ((tt, (Y1, Y2, Y3), 'tt'), (tA, (Y0, Y2, Y3), 'tA'), (thth, (Y1, Y3), 'thth')):
            for s in bad:
                c = co(e, s)
                if c != 0:
                    assert _numcheck(c, {th: sp.Rational(9, 10)}) < 1e-25, (nm, s)
        out['tt'] = _thetafree(co(tt, Y0), 'tt')
        out['tA'] = _thetafree(co(tA, Y1), 'tA')
        Atf = _thetafree(co(thth, Y2), 'tf')
        Atr = _thetafree(co(thth, Y0), 'tr0') - lam / 2 * Atf
        out['tr'], out['tf'] = Atr, Atf
        # consistency of the phi-phi component
        chk = phph / sp.sin(th)**2 - (Atr * Y0 + Atf * (sp.cos(th) / sp.sin(th) * Y1 + lam / 2 * Y0))
        chk = sp.expand(chk)
        for s in (Y0, Y1, Y2, Y3):
            c = co(chk, s)
            if c != 0:
                assert _numcheck(c, {th: sp.Rational(9, 10)}) < 1e-25, ('phph', s)
        out['_zero'] = 0
    else:
        tph, thph = sym(T[0, 2]), sym(T[1, 2])
        out['tA'] = _thetafree(co(tph, Y1) / sp.sin(th), 'tA')
        AAB = _thetafree(2 * co(thph, Y2) / sp.sin(th), 'AB')
        out['AB'] = AAB
        for e, bad in ((tph, (Y0, Y2, Y3)), (thph, (Y0, Y3))):
            for s in bad:
                c = co(e, s)
                if c != 0:
                    assert _numcheck(c, {th: sp.Rational(9, 10)}) < 1e-25, ('odd', s)
        c = co(thph, Y1) + AAB * sp.sin(th) / 2 * sp.cos(th) / sp.sin(th)
        if c != 0:
            assert _numcheck(sp.expand(c), {th: sp.Rational(9, 10)}) < 1e-25, 'odd thph Y1'
        for (i, j) in [(0, 0), (0, 1), (1, 1), (2, 2)]:
            e = sym(T[i, j])
            if e != 0:
                assert _numcheck(e, {th: sp.Rational(9, 10)}) < 1e-25, (i, j)
    out.pop('_zero', None)
    return out


def E_components(ch, cK, h0, K0, parity):
    """Harmonic coefficients of E_ab = K_ab - h_ab K at first order, from those of
    delta h_ab and delta K_ab (background h0 = diag(-1, R^2, R^2 sin^2)):
      dK_trace = -K_tt + 2 K_tr/R^2 - h_tt K0_tt - 2 K0_thth h_tr / R^4
      E_tt = K_tt - h_tt K0 + dK_trace ;  E_tA = K_tA - h_tA K0
      E_tr = K_tr - h_tr K0 - R^2 dK_trace ;  E_tf = K_tf - h_tf K0
    (odd: E_tA = K_tA - h_tA K0, E_AB = K_AB - h_AB K0; no odd trace)."""
    K0tt, K0thth = K0[0, 0], K0[1, 1]
    R2_ = h0[1, 1]
    K0tr = -K0tt + 2 * K0thth / R2_
    if parity == 'even':
        dKtr = -cK['tt'] + 2 * cK['tr'] / R2_ - ch['tt'] * K0tt - 2 * K0thth * ch['tr'] / R2_**2
        return {'tt': cK['tt'] - ch['tt'] * K0tr + dKtr,
                'tA': cK['tA'] - ch['tA'] * K0tr,
                'tr': cK['tr'] - ch['tr'] * K0tr - R2_ * dKtr,
                'tf': cK['tf'] - ch['tf'] * K0tr}
    return {'tA': cK['tA'] - ch['tA'] * K0tr, 'AB': cK['AB'] - ch['AB'] * K0tr}


def trace_part(h0, K0, h1, K1):
    """first-order part of E_ab = K_ab - h_ab K,  K = h^{ab} K_ab."""
    hinv0 = h0.inv()
    Ktr0 = sum(hinv0[a, b] * K0[a, b] for a in range(3) for b in range(3))
    hinv1 = -hinv0 * h1 * hinv0
    Ktr1 = sum(hinv1[a, b] * K0[a, b] + hinv0[a, b] * K1[a, b] for a in range(3) for b in range(3))
    E0 = K0 - h0 * Ktr0
    E1 = K1 - h1 * Ktr0 - h0 * Ktr1
    return E0, E1, Ktr0, Ktr1


def build_side(side, parity, dynamic):
    """Geometry of one side, frozen and in terms of bulk symbols."""
    F = 1 - 2 * M / r if side == '+' else sp.Integer(1)
    if parity == 'even':
        fz = {'xi': sp.Function('xi' + ('p' if side == '+' else 'm'))(tau)}
        if side == '-':
            fz['zeta'] = sp.Function('zeta')(tau)
            fz['eta'] = sp.Function('eta')(tau)
    else:
        fz = {'chi': sp.Function('chi')(tau)} if side == '-' else {}
    geo = side_geometry(F, parity, fz)
    FR = F.subs(r, Rs)
    nu = w / sp.sqrt(1 - 2 * M / Rs)          # proper-time frequency on the shell
    ws = w if side == '+' else nu              # bulk frequency in that side's time
    out = {}
    for k in ('h0', 'h1', 'K0', 'K1'):
        m = geo[k].applyfunc(lambda e: freeze(e, F, dynamic, nu))
        m = m.applyfunc(lambda e: time_harmonic(e, list(fz.values()), nu))
        m = m.applyfunc(lambda e: e.subs(geo['ws'], ws))
        # bulk time derivative factors: exp(-i ws T0) already dropped; nothing else
        out[k] = m.applyfunc(lambda e: sp.expand(e))
        if k in ('h0', 'K0'):
            out[k] = out[k].applyfunc(cz)
    out['bulk'] = geo['bulk']
    out['nu'] = nu
    return out


def bulk_to_master(expr, side, parity):
    """Replace RW-gauge bulk values at r=R by master function values
    Psi = P, dPsi/dr = dP on that side, using vacuum_relations.pkl."""
    Ms = M if side == '+' else 0
    ws = w if side == '+' else w / sp.sqrt(1 - 2 * M / Rs)
    Pn, dPn = sp.symbols('P' + ('p' if side == '+' else 'm') + ' dP' + ('p' if side == '+' else 'm'))
    Psi = sp.Function('Psi')(rv)
    f_ = 1 - 2 * Mv / rv
    V = VAC['V']['RW'] if parity == 'odd' else VAC['V']['Z']

    def psi2(e):
        dP = sp.diff(Psi, rv)
        p2 = ((V - wv**2) * Psi / f_ - sp.diff(f_, rv) * dP) / f_
        for _ in range(2):
            e = e.subs(sp.Derivative(Psi, (rv, 3)), sp.diff(p2, rv))
            e = e.subs(sp.Derivative(Psi, (rv, 2)), p2)
        return e

    if parity == 'odd':
        rel = {'h0': VAC['odd']['ht'], 'h1': VAC['odd']['hr']}
    else:
        P, dP = VAC['even']['P'], VAC['even']['dP']
        rel = {k: VAC['even'][k].subs({P: Psi, dP: sp.diff(Psi, rv)}) for k in ('H', 'H1', 'K')}
    subs = {}
    for nm, e in rel.items():
        for suffix, ee in (('', e), ('_r', sp.diff(e, rv)), ('_rr', sp.diff(e, rv, 2))):
            val = psi2(ee)
            val = val.subs(sp.Derivative(Psi, rv), dPn).subs(Psi, Pn)
            val = val.subs({Mv: Ms, wv: ws, rv: Rs})
            subs[sp.Symbol(nm + suffix)] = val
    return expr.subs(subs), Pn, dPn


def derive(parity, model, vs2=sp.Symbol('v_s2')):
    """Build the linear junction system  A x = B (Psi_-, dPsi_-/dr)  with
    x = (shell displacement / twist variables, shell-matter variables, Psi_+, dPsi_+/dr)."""
    dynamic = (model == 'A')
    sides = {s: build_side(s, parity, dynamic) for s in ('+', '-')}
    Mat = {}
    for s in ('+', '-'):
        g = sides[s]
        E0, E1, K0tr, K1tr = trace_part(g['h0'], g['K0'], g['h1'], g['K1'])
        comp_h = harmonic_components(g['h1'], parity)
        comp_K = harmonic_components(g['K1'], parity)
        comp_E = E_components(comp_h, comp_K, g['h0'], g['K0'], parity)
        conv = {}
        for k in comp_h:
            conv[('h', k)], Pn, dPn = bulk_to_master(comp_h[k], s, parity)
        for k in comp_E:
            conv[('E', k)], Pn, dPn = bulk_to_master(comp_E[k], s, parity)
        Mat[s] = dict(conv=conv, E0=E0.applyfunc(cz), h0=g['h0'], K0tr=cz(K0tr),
                      P=Pn, dP=dPn, nu=g['nu'], comp_h=comp_h, comp_E=comp_E)
        print('   side', s, 'projected', flush=True)
    sq = sp.sqrt(1 - 2 * M / Rs)
    sig = (1 - sq) / (4 * sp.pi * Rs)                            # Poisson (2004) Eq. (3.80)
    if model == 'C':
        p = (1 - M / Rs - sq) / (8 * sp.pi * Rs * sq)            # Poisson (2004) Eq. (3.81)
    else:
        p = -sig                                                 # domain wall, tau = sigma (I&S 2.10)
    h0 = Mat['+']['h0']
    u0 = sp.Matrix([-1, 0, 0])
    S0 = (sig + p) * u0 * u0.T + p * h0
    bg = (Mat['+']['E0'] - Mat['-']['E0'] + 8 * sp.pi * S0).applyfunc(cz)
    info = {'background_residual': bg, 'sigma': sig, 'p': p}
    if model == 'A':
        # tau-tau background Israel equation fixes Rddot (Ipser & Sikivie Eq. 3.7a)
        # (E_tautau = 2 K^theta_theta carries no Rddot; the angular block does)
        comp = next(bg[i, i] for i in range(3) if bg[i, i].has(Rdd))
        Rdd_sol = sp.solve(comp, Rdd)[0]
        info['Rdd'] = cz(Rdd_sol)
        bg = bg.applyfunc(lambda e: cz(e.subs(Rdd, Rdd_sol)))
        info['background_residual_after_Rdd'] = bg
    dsig, U = sp.symbols('dsig U')
    hp, hm = Mat['+']['conv'], Mat['-']['conv']
    comps = ['tt', 'tA', 'tr', 'tf'] if parity == 'even' else ['tA', 'AB']
    eqs, labels = [], []
    for k in comps:
        eqs.append(sp.expand(hp[('h', k)] - hm[('h', k)]))                  # [h_ab] = 0   (Poisson 3.49)
        labels.append('[h_%s]' % k)
    unknown_matter = []
    Sc = {}
    if model == 'C':
        if parity == 'even':
            htt, htA, htr, htf = (hp[('h', k)] for k in comps)
            # S_ab = (sig+p) u_a u_b + p h_ab, u_tau = -sqrt(-h_tautau), u_A = eps U D_A Y,
            # sig -> sig + eps dsig Y, p -> p + eps vs2 dsig Y     (first-order coefficients)
            Sc = dict(tt=dsig - sig * htt, tA=-(sig + p) * U + p * htA,
                      tr=vs2 * dsig * Rs**2 + p * htr, tf=p * htf)
            unknown_matter = [dsig, U]
        else:
            htA, hAB = hp[('h', 'tA')], hp[('h', 'AB')]
            Sc = dict(tA=-(sig + p) * U + p * htA, AB=p * hAB)
            unknown_matter = [U]
    else:  # domain wall: S_ab = -sig h_ab with sig constant (I&S Eqs. 2.10, 2.12)
        for k in comps:
            Sc[k] = -sig * hp[('h', k)]
    for k in comps:
        e = hp[('E', k)] - hm[('E', k)] + 8 * sp.pi * Sc[k]         # [K_ab - h_ab K] = -8 pi S_ab
        if model == 'A':
            e = e.subs(Rdd, info['Rdd'])
        eqs.append(sp.expand(e))
        labels.append('Israel_%s' % k)
    if parity == 'even':
        shell = [sp.Symbol('xip_a'), sp.Symbol('xim_a'), sp.Symbol('zeta_a'), sp.Symbol('eta_a')]
    else:
        shell = [sp.Symbol('chi_a')]
    Pp, dPp, Pm, dPm = Mat['+']['P'], Mat['+']['dP'], Mat['-']['P'], Mat['-']['dP']
    unknowns = shell + unknown_matter + [Pp, dPp]
    A_, rhs = sp.linear_eq_to_matrix(eqs, unknowns)
    B_ = sp.Matrix([[sp.diff(e, Pm), sp.diff(e, dPm)] for e in rhs])
    info.update(dict(A=A_, B=B_, unknowns=unknowns, labels=labels, eqs=eqs,
                     Pm=Pm, dPm=dPm, vs2=vs2, comps_side={s: (Mat[s]['comp_h'], Mat[s]['comp_E'])
                                                           for s in Mat}))
    return info


def numeric_T(info, vals):
    """Solve A x = B y numerically (least squares if overdetermined) -> T (2x2),
    ranks, and the residual of the (possibly overdetermined) system."""
    import numpy as np
    A = np.array(info['A'].subs(vals).evalf(), dtype=complex)
    B = np.array(info['B'].subs(vals).evalf(), dtype=complex)
    rA = np.linalg.matrix_rank(A, tol=1e-9 * np.abs(A).max())
    rAB = np.linalg.matrix_rank(np.hstack([A, B]), tol=1e-9 * np.abs(A).max())
    X, res, rk, sv = np.linalg.lstsq(A, B, rcond=None)
    resid = np.abs(A @ X - B).max() / np.abs(B).max()
    return X[-2:, :], rA, rAB, resid


def export_module(info, name):
    """Write shellgw/junction_even_<model>.py (or odd) with numeric A(w,lam,R,M,vs2), B(...)."""
    import os
    args = [w, lam, Rs, M, info['vs2']]
    code = ['"""AUTO-GENERATED by code/symbolic/derive_junction.py -- do not edit.',
            'Linear junction system A x = B (Psi_-, dPsi_-/dr), unknowns:',
            '  ' + ', '.join(str(u) for u in info['unknowns']),
            'rows: ' + ', '.join(info['labels']),
            'T = last two rows of A^+ B (least squares if overdetermined)."""',
            'import numpy as np', 'from numpy import sqrt, pi, exp', '',
            'I = 1j', '', '']
    for nm, Mx in (('A', info['A']), ('B', info['B'])):
        code.append('def %s(omega, lam, R, M, v_s2):' % nm)
        rows = []
        for i in range(Mx.shape[0]):
            rows.append('[' + ', '.join(sp.pycode(Mx[i, j]).replace('math.', '').replace('lambda_', 'lam') for j in range(Mx.shape[1])) + ']')
        code.append('    return np.array([' + ',\n        '.join(rows) + '], dtype=complex)')
        code.append('')
    code.append('def T(omega, lam, R, M=1.0, v_s2=0.0):')
    code.append('    a = A(omega, lam, R, M, v_s2)')
    code.append('    b = B(omega, lam, R, M, v_s2)')
    code.append('    X = np.linalg.lstsq(a, b, rcond=None)[0]')
    code.append('    return X[-2:, :]')
    code.append('')
    code.append('def residual(omega, lam, R, M=1.0, v_s2=0.0):')
    code.append('    a = A(omega, lam, R, M, v_s2); b = B(omega, lam, R, M, v_s2)')
    code.append('    X = np.linalg.lstsq(a, b, rcond=None)[0]')
    code.append('    return np.abs(a @ X - b).max() / np.abs(b).max()')
    path = os.path.join(os.path.dirname(__file__), '..', 'shellgw', name + '.py')
    with open(path, 'w', encoding='utf8') as fh:
        fh.write('\n'.join(code) + '\n')
    return path


if __name__ == '__main__':
    import sys
    jobs = [('odd', 'C'), ('odd', 'A'), ('even', 'C'), ('even', 'A')]
    if len(sys.argv) > 1:
        jobs = [tuple(a.split(':')) for a in sys.argv[1:]]
    test_vals = {w: sp.Float(0.37) - sp.Float(0.05) * sp.I, lam: 6, Rs: sp.Float(3.7), M: 1,
                 sp.Symbol('v_s2'): sp.Float(0.3)}
    for parity, model in jobs:
        print(f'=== parity={parity} model={model}', flush=True)
        info = derive(parity, model)
        print('  background residual:', info['background_residual'].applyfunc(sp.simplify))
        if 'Rdd' in info:
            print('  Rddot =', sp.simplify(info['Rdd']))
            print('  background residual after Rdd:', info['background_residual_after_Rdd'].applyfunc(sp.simplify))
        Tn, rA, rAB, resid = numeric_T(info, test_vals)
        print('  unknowns:', info['unknowns'])
        print('  n_eqs=%d n_unknowns=%d rank(A)=%d rank([A|B])=%d residual=%.2e'
              % (len(info['eqs']), len(info['unknowns']), rA, rAB, resid))
        print('  T(test) =', Tn)
        if parity == 'odd':
            sol = sp.solve(info['eqs'], info['unknowns'], dict=True)
            if sol:
                s0 = sol[0]
                print('  symbolic solution:', {k: sp.simplify(v) for k, v in s0.items()})
            else:
                # overdetermined: solve all but check consistency
                sol = sp.solve(info['eqs'][:len(info['unknowns'])], info['unknowns'], dict=True)[0]
                rest = [sp.simplify(e.subs(sol)) for e in info['eqs'][len(info['unknowns']):]]
                print('  symbolic solution (first n eqs):', {k: sp.simplify(v) for k, v in sol.items()},
                      ' remaining eqs ->', rest)
        path = export_module(info, 'junction_%s_%s' % (parity, model))
        print('  exported', path, flush=True)
        with open('junction_%s_%s.pkl' % (parity, model), 'wb') as fh:
            pickle.dump({k: v for k, v in info.items() if k != 'comps_side'}, fh)



