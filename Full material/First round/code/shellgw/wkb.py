"""
WKB quasinormal-mode formulas of arbitrary order.

The N-th order WKB formula of Schutz & Will (1985) [N=1], Iyer & Will (1987)
[N=3, their Eqs. (1.5)] and Konoplya (2003) [N=6] is the truncation of the
perturbative (Rayleigh-Schroedinger / Bender-Wu) expansion of the eigenvalue of
the inverted anharmonic oscillator obtained by Taylor expanding the potential
about its maximum (see e.g. Hatsuda, PRD 101, 024008 (2020), Sec. II).  We
generate that expansion exactly, with symbolic overtone n, using the
hypervirial-Hellmann-Feynman recursion (Killingbeck 1978; Fernandez & Castro
1987) for H = -d^2/dx^2 + V(x):

   2k E <x^{k-1}> = 2k <x^{k-1} V> + <x^k V'> - 1/2 k(k-1)(k-2) <x^{k-3}>.

With V - V0 = a x^2 + sum_{j>=3} g^{j-2} v_j x^j and E = w^2 - V0, the terms
through g^{2(N-1)} give the N-th order WKB formula.  The root sqrt(a) is taken
as -i sqrt(-V0''/2), which reproduces w^2 = V0 - i(n+1/2) sqrt(-2 V0'') at
leading order (Schutz & Will 1985, Eq. 7).

The code reproduces Iyer & Will (1987) Table I at third order (see tests in
main.py), which validates the construction.
"""
from functools import lru_cache
import numpy as np
import sympy as sp


@lru_cache(maxsize=None)
def _hpm_energy(order):
    """Symbolic E(n, a, v3..v_{2N}) through g^{2(order-1)}."""
    n = sp.Symbol('n')
    a = sp.Symbol('a')           # coefficient of x^2 (sqrt(a) kept as symbol s)
    s = sp.Symbol('s')           # s = sqrt(a)
    gmax = 2 * (order - 1)
    kmax = gmax + 2
    v = {j: sp.Symbol('v%d' % j) for j in range(3, kmax + 1)}
    # perturbative orders m = 0..gmax ; potential coefficient of x^j carries g^(j-2)
    # unknowns: E[m], X[k][m]
    E = [None] * (gmax + 1)
    E[0] = s * (2 * n + 1)
    # X_k^{(m)}; X_0 = 1 exactly
    X = {}

    def getX(k, m):
        if k < 0:
            return 0
        if k == 0:
            return 1 if m == 0 else 0
        return X[(k, m)]

    # moments needed: recursion in k at fixed m requires lower m for anharmonic terms
    for m in range(gmax + 1):
        if m > 0:
            # Hellmann-Feynman (dE/dg = sum (j-2) g^{j-3} v_j <x^j>) -> E^{(m)} m = sum_j (j-2) v_j X_j^{(m-(j-2))}
            acc = 0
            for j in range(3, kmax + 1):
                mm = m - (j - 2)
                if mm >= 0:
                    acc += (j - 2) * v[j] * getX(j, mm)
            E[m] = sp.expand(acc / m)
        # now compute X_k^{(m)} for k up to needed range, using hypervirial with k -> k
        # 2k sum_{m'} E^{(m')} X_{k-1}^{(m-m')} = 2k [a X_{k+1}^{(m)} + sum_j v_j X_{k+j-1}^{(m-(j-2))}]
        #   + [2 a X_{k+1}^{(m)} + sum_j j v_j X_{k+j-1}^{(m-(j-2))}] - 1/2 k(k-1)(k-2) X_{k-3}^{(m)}
        kneed = kmax * (gmax - m + 1) + 2
        # k = 0 hypervirial relation <V'> = 0 gives X_1:  2a X_1 + sum_j j v_j X_{j-1} = 0
        acc = 0
        for j in range(3, kmax + 1):
            mm = m - (j - 2)
            if mm >= 0:
                acc += j * v[j] * getX(j - 1, mm)
        X[(1, m)] = sp.expand(-acc / (2 * s**2))
        for k in range(1, kneed + 1):
            lhs = 0
            for mp in range(m + 1):
                if k - 1 == 0:
                    lhs += 2 * k * E[mp] * (1 if m - mp == 0 else 0)
                elif (k - 1, m - mp) in X:
                    lhs += 2 * k * E[mp] * X[(k - 1, m - mp)]
            rest = 0
            for j in range(3, kmax + 1):
                mm = m - (j - 2)
                if mm >= 0:
                    rest += (2 * k + j) * v[j] * getX(k + j - 1, mm)
            rest -= sp.Rational(1, 2) * k * (k - 1) * (k - 2) * getX(k - 3, m)
            # (2k+2) a X_{k+1}^{(m)} = lhs - rest
            X[(k + 1, m)] = sp.expand((lhs - rest) / ((2 * k + 2) * s**2))
    Etot = sum(E)
    return Etot, n, s, v


@lru_cache(maxsize=None)
def _energy_func(order):
    """Lambdified E(n, s, v3..v_{2N}); cached on disk (symbolic build is slow at N=6)."""
    import os
    import pickle
    cache = os.path.join(os.path.dirname(__file__), '_wkb_cache_%d.pkl' % order)
    if os.path.exists(cache):
        with open(cache, 'rb') as fh:
            Etot, nn, s, v = pickle.load(fh)
    else:
        Etot, nn, s, v = _hpm_energy(order)
        with open(cache, 'wb') as fh:
            pickle.dump((Etot, nn, s, v), fh)
    keys = sorted(v)
    fn = sp.lambdify([nn, s] + [v[k] for k in keys], Etot, 'numpy')
    return fn, keys


def trapped_modes_bs(l, parity, R, nmax=10, M=1.0):
    """WKB estimate of the long-lived modes trapped between the flat interior and
    the exterior potential barrier of an ultracompact shell (R < 3M).

    Real part (Bohr-Sommerfeld with Langer's l(l+1) -> (l+1/2)^2 in the interior):
        Phi(w) = int_{r1}^{R} sqrt(nu^2 - (l+1/2)^2/r^2) dr
               + int_R^{r_a} sqrt(w^2 - V(r)) dr/f  = pi (n + 1/2),
        nu = w/sqrt(f(R)), r1 = (l+1/2)/nu, r_a inner turning point of the barrier.
    Imaginary part (Gamow): Im w = -T_b / (4 w int dx / sqrt(w^2 - V)),
        T_b = exp(-2 int_{r_a}^{r_b} sqrt(V - w^2) dr/f).
    The delta-function shell coupling (strength sqrt f (1-sqrt f)/R -> 0 as R -> 2M)
    is neglected; accuracy is expected to improve as R -> 2M and for T_b << 1.
    """
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from .potentials import V as Vp, peak_derivatives
    sf = np.sqrt(1 - 2 * M / R)
    rpk, d = peak_derivatives(l, parity, 2)
    Vmax = d[0]
    L2 = (l + 0.5)**2

    def turning(w):
        g = lambda r: Vp(r, l, parity, M) - w**2
        if R < rpk and g(R) < 0:
            ra = brentq(g, R, rpk)
        else:
            ra = R
        rb = brentq(g, rpk, 1e4)
        return ra, rb

    def Phi(w):
        nu = w / sf
        r1 = np.sqrt(L2) / nu
        inner = quad(lambda r: np.sqrt(max(nu**2 - L2 / r**2, 0.0)), min(r1, R), R, limit=200)[0]
        ra, rb = turning(w)
        outer = quad(lambda r: np.sqrt(max(w**2 - Vp(r, l, parity, M), 0.0)) / (1 - 2 * M / r),
                     R, ra, limit=200)[0] if ra > R else 0.0
        return inner + outer

    modes = []
    ws = np.linspace(1e-3, np.sqrt(Vmax) * 0.999, 4000)
    ph = np.array([Phi(w) for w in ws])
    for n in range(nmax):
        target = np.pi * (n + 0.5)
        idx = np.where((ph[:-1] - target) * (ph[1:] - target) < 0)[0]
        if len(idx) == 0:
            break
        i = idx[0]
        wR = brentq(lambda w: Phi(w) - target, ws[i], ws[i + 1])
        ra, rb = turning(wR)
        Tb = np.exp(-2 * quad(lambda r: np.sqrt(max(Vp(r, l, parity, M) - wR**2, 0.0)) / (1 - 2 * M / r),
                              ra, rb, limit=200)[0])
        nu = wR / sf
        r1 = np.sqrt(L2) / nu
        # int dx / sqrt(w^2 - V): inside dx = dr/sqrt(f_R), sqrt(w^2-V) = sqrt(f_R) sqrt(nu^2 - L2/r^2)
        tin = quad(lambda r: 1 / np.sqrt(max(nu**2 - L2 / r**2, 1e-30)), min(r1, R) * (1 + 1e-9), R,
                   limit=200)[0] / sf**2
        tout = quad(lambda r: 1 / np.sqrt(max(wR**2 - Vp(r, l, parity, M), 1e-30)) / (1 - 2 * M / r),
                    R, ra * (1 - 1e-9), limit=200)[0] if ra > R else 0.0
        tint = tin + tout
        wI = -Tb / (4 * wR * tint)
        modes.append(wR + 1j * wI)
    return modes


def wkb_qnm(Vders, n, order):
    """QNM frequency from WKB of given order.

    Vders: list [V0, V1, ..., V_{2*order}] of derivatives of V w.r.t. r* at the
           maximum (V1 ~ 0).  Returns omega with Re>0, Im<0 (e^{-i w t}).
    Formula: w^2 = V0 + E(n) with E from the hypervirial expansion
             (equivalent to Iyer & Will 1987 Eq. 1.5 at order 3).
    """
    fn, keys = _energy_func(order)
    V2 = Vders[2]
    sval = -1j * np.sqrt(-V2 / 2.0)
    args = [n, sval] + [Vders[j] / float(sp.factorial(j)) for j in keys]
    E = complex(fn(*args))
    w2 = Vders[0] + E
    w = np.sqrt(w2 + 0j)
    if w.real < 0:
        w = -w
    return w
