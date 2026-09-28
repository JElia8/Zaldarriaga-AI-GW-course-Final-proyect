"""
Flat interior (0 <= r < R).

Both master functions obey  Psi'' + [nu^2 - l(l+1)/r^2] Psi = 0
(interior_ondas_cascara Eqs. (8)-(9); Martel & Poisson 2005 Eqs. (4.25), (5.14)
with M = 0), where nu is the frequency conjugate to the *interior* Minkowski time T.

Frequency normalisation (important, not in the prior notes): with a common
Killing time t (exterior Schwarzschild time), continuity of the induced metric
at a static shell requires dT = sqrt(f(R)) dt (Poisson 2004, Eq. 3.75), hence
    nu = omega / sqrt(f(R)).
The interior wave is blue-shifted relative to the exterior one.

Riccati-Bessel functions (Abramowitz & Stegun 10.3.1; interior_ondas_cascara
Eqs. (10), (26)):  jhat_l(x) = x j_l(x) (regular, ~ x^{l+1}/(2l+1)!!),
yhat_l(x) = x y_l(x).  Riccati-Hankel (e^{-i w t} convention):
    hout_l = -yhat + i jhat ~ e^{+i(x - l pi/2)}  (outgoing),
    hin_l  = -yhat - i jhat ~ e^{-i(x - l pi/2)}  (ingoing).
Wronskian: jhat yhat' - jhat' yhat = 1 (A&S 10.3.? / DLMF 10.50.1).
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn


def jhat(l, x):
    """Riccati-Bessel x j_l(x) and its derivative; complex x allowed."""
    j = spherical_jn(l, x)
    jp = spherical_jn(l, x, derivative=True)
    return x * j, j + x * jp


def yhat(l, x):
    """Riccati-Bessel x y_l(x) and its derivative; complex x allowed."""
    y = spherical_yn(l, x)
    yp = spherical_yn(l, x, derivative=True)
    return x * y, y + x * yp


def hin(l, x):
    y, yp = yhat(l, x)
    j, jp = jhat(l, x)
    return -y - 1j * j, -yp - 1j * jp


def hout(l, x):
    y, yp = yhat(l, x)
    j, jp = jhat(l, x)
    return -y + 1j * j, -yp + 1j * jp


def nu_of(omega, R, M=1.0):
    """Interior frequency nu = omega / sqrt(1 - 2M/R)."""
    return omega / np.sqrt(1.0 - 2.0 * M / R)


def regular_at_shell(omega, l, R, M=1.0):
    """(Psi, dPsi/dr) at r = R^- of the regular interior solution jhat_l(nu r)
    [interior_ondas_cascara Eq. (13), with nu instead of omega]."""
    nu = nu_of(omega, R, M)
    j, jp = jhat(l, nu * R)
    return np.array([j, nu * jp])


def ingoing_at_shell(omega, l, R, M=1.0):
    """(Psi, dPsi/dr) at r = R^- of the ingoing interior wave hin_l(nu r)."""
    nu = nu_of(omega, R, M)
    h, hp = hin(l, nu * R)
    return np.array([h, nu * hp])


def regular_numerical(omega, l, R, M=1.0, r0_frac=1e-3):
    """Independent check: integrate the interior equation from r0 << R with the
    power-series start Psi ~ r^{l+1}(1 - (nu r)^2/(2(2l+3)) + ...) using DOP853
    (normalisation matched to jhat_l(nu r) ~ (nu r)^{l+1}/(2l+1)!!)."""
    from scipy.integrate import solve_ivp
    from scipy.special import factorial2
    nu = nu_of(omega, R, M)
    r0 = r0_frac * R
    x0 = nu * r0
    c0 = 1.0 / factorial2(2 * l + 1)
    # two-term series of jhat: x^{l+1}/(2l+1)!! [1 - x^2/(2(2l+3)) + x^4/(8(2l+3)(2l+5))]
    s = 1 - x0**2 / (2 * (2 * l + 3)) + x0**4 / (8 * (2 * l + 3) * (2 * l + 5))
    ds = -2 * x0 / (2 * (2 * l + 3)) + 4 * x0**3 / (8 * (2 * l + 3) * (2 * l + 5))
    P0 = c0 * x0**(l + 1) * s
    dP0 = nu * c0 * ((l + 1) * x0**l * s + x0**(l + 1) * ds)

    def rhs(r, y):
        return [y[1], (l * (l + 1) / r**2 - nu**2) * y[0]]

    sol = solve_ivp(rhs, (r0, R), np.array([P0, dP0], dtype=complex), method='DOP853',
                    rtol=1e-10, atol=1e-12 * abs(P0) + 1e-300)
    return sol.y[:, -1]
