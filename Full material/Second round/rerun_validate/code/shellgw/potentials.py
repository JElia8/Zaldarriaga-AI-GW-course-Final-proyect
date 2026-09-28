"""
Effective potentials and tortoise coordinate (units: G = c = 1, M = 1 unless stated).

  V_RW = f [ l(l+1)/r^2 - 6M/r^3 ]                 Regge & Wheeler (1957) Eq. (25);
                                                     Martel & Poisson (2005) Eq. (5.15) x f
  V_Z  = f [2n^2(n+1)r^3 + 6n^2 M r^2 + 18 n M^2 r + 18 M^3] / [r^3 (n r + 3M)^2],
         n = (l-1)(l+2)/2                           Zerilli (1970) Eq. (5);
                                                     = f x Martel & Poisson (2005) Eq. (4.26)
                                                     (identity verified in symbolic/vacuum_relations.py)
  r*   = r + 2M ln(r/2M - 1)                        Zerilli (1970) Eq. (3)
  interior (flat): V_int = l(l+1)/r^2               interior_ondas_cascara Eqs. (6)-(8)
"""
from functools import lru_cache
import numpy as np
import sympy as sp


def f(r, M=1.0):
    return 1.0 - 2.0 * M / r


def V_RW(r, l, M=1.0):
    """Regge-Wheeler potential, RW (1957) Eq. (25) / MP05 Eq. (5.15)."""
    return f(r, M) * (l * (l + 1) / r**2 - 6.0 * M / r**3)


def V_Z(r, l, M=1.0):
    """Zerilli potential, Zerilli (1970) Eq. (5) / MP05 Eq. (4.26)."""
    n = (l - 1) * (l + 2) / 2.0
    num = 2 * n**2 * (n + 1) * r**3 + 6 * n**2 * M * r**2 + 18 * n * M**2 * r + 18 * M**3
    return f(r, M) * num / (r**3 * (n * r + 3 * M)**2)


def V(r, l, parity, M=1.0):
    return V_RW(r, l, M) if parity == 'odd' else V_Z(r, l, M)


def rstar(r, M=1.0):
    """Tortoise coordinate, Zerilli (1970) Eq. (3)."""
    return r + 2.0 * M * np.log(r / (2.0 * M) - 1.0)


def r_of_rstar(x, M=1.0):
    """Invert r*(r) via Lambert W: r = 2M [1 + W(exp(x/2M - 1))]."""
    from scipy.special import lambertw
    return 2.0 * M * (1.0 + np.real(lambertw(np.exp(x / (2.0 * M) - 1.0))))


@lru_cache(maxsize=None)
def _sym_derivs(l, parity, nmax):
    r = sp.Symbol('r', positive=True)
    M = 1
    fr = 1 - 2 * M / r
    if parity == 'odd':
        Vs = fr * (l * (l + 1) / r**2 - sp.Rational(6) * M / r**3)
    else:
        n = sp.Rational((l - 1) * (l + 2), 2)
        Vs = fr * (2 * n**2 * (n + 1) * r**3 + 6 * n**2 * M * r**2 + 18 * n * M**2 * r + 18 * M**3) \
            / (r**3 * (n * r + 3 * M)**2)
    ders = [Vs]
    for _ in range(nmax):
        ders.append(sp.simplify(fr * sp.diff(ders[-1], r)))      # d/dr* = f d/dr
    return [sp.lambdify(r, d, 'mpmath') for d in ders]


def peak_derivatives(l, parity, nmax):
    """Location of the potential maximum and d^k V/dr*^k there (k=0..nmax), M = 1."""
    import mpmath as mp
    mp.mp.dps = 40
    fs = _sym_derivs(l, parity, nmax)
    r0 = mp.findroot(fs[1], 3.0)
    return float(r0), [float(fk(r0)) for fk in fs]
