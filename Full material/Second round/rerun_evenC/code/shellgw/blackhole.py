"""
Pure Schwarzschild black hole: validation of the exterior machinery.

 * QNMs by Leaver's continued fraction (Leaver 1985, Proc. R. Soc. A 402, 285,
   Eqs. (5)-(8)), units 2M = 1, spin s = 2 (RW); Zerilli QNMs are identical
   (Chandrasekhar isospectrality).
       alpha_n = n^2 + (2 rho + 2) n + 2 rho + 1
       beta_n  = -[2 n^2 + (8 rho + 2) n + 8 rho^2 + 4 rho + l(l+1) - s^2 + 1]
       gamma_n = n^2 + 4 rho n + 4 rho^2 - s^2,     rho = -i w (2M = 1).
 * Reflection coefficient |B_out/B_in|^2 with ingoing horizon condition
   Psi ~ e^{-i w r*} (r -> 2M), obtained by DOP853 integration from near the
   horizon (series start) - the quantity plotted by Vishveshwara (1970) Fig. 2
   (R vs k^2 = (2 M w)^2 for l = 2 odd parity).
"""
import numpy as np
from scipy.optimize import root
from .exterior import integrate_real, up_asymptotic


def leaver_cf(w, l, N=500, s=2, inv=0):
    """Leaver continued-fraction function with inversion index inv (zero at QNMs);
    the inv-th inversion is best conditioned for the inv-th overtone (Leaver 1985 Sec. 4):
      f_n = beta_n - a_{n-1} g_n/(beta_{n-1} - a_{n-2} g_{n-1}/(... beta_0))
                   - a_n g_{n+1}/(beta_{n+1} - a_{n+1} g_{n+2}/(beta_{n+2} - ...)).
    w in units M = 1."""
    wl = 2.0 * w                 # 2 M w
    rho = -1j * wl
    n = np.arange(N + 1)
    al = n**2 + (2 * rho + 2) * n + 2 * rho + 1
    be = -(2 * n**2 + (8 * rho + 2) * n + 8 * rho**2 + 4 * rho + l * (l + 1) - s**2 + 1)
    ga = n**2 + 4 * rho * n + 4 * rho**2 - s**2
    tail = 0.0 + 0j
    for k in range(N, inv, -1):
        tail = al[k - 1] * ga[k] / (be[k] - tail)
    head = 0.0 + 0j
    if inv > 0:
        head = be[0]
        for k in range(1, inv):
            head = be[k] - al[k - 1] * ga[k] / head
        head = al[inv - 1] * ga[inv] / head
    return be[inv] - head - tail


def leaver_qnm(l, guess, N=800, inv=0):
    """Solve leaver_cf = 0 (scipy.optimize.root on real/imag parts); returns the root
    and its change when the continued-fraction depth is doubled."""
    def F(x, NN):
        v = leaver_cf(x[0] + 1j * x[1], l, NN, inv=inv)
        return [v.real, v.imag]
    sol = root(F, [guess.real, guess.imag], args=(N,), tol=1e-14)
    sol2 = root(F, sol.x, args=(2 * N,), tol=1e-14)
    w = sol.x[0] + 1j * sol.x[1]
    w2 = sol2.x[0] + 1j * sol2.x[1]
    return w2, abs(w2 - w)


def horizon_start(w, l, parity, M=1.0, eps=1e-6):
    """(Psi, dPsi/dr) of the horizon-ingoing solution at r = 2M(1+eps):
    Psi = e^{-i w r*} (1 + O(r-2M)), first-order Frobenius correction included."""
    r = 2 * M * (1 + eps)
    x = r - 2 * M
    rs = r + 2 * M * np.log(x / (2 * M))
    # Psi = e^{-i w r*} u,  f u'' + (f' - 2 i w) u' - (V/f) u = 0  ->  at r = 2M:
    # u'/u = a1 = (V/f)|_{2M} / (1/(2M) - 2 i w)
    rh = 2 * M
    lam = l * (l + 1)
    if parity == 'odd':
        q = lam / rh**2 - 6 * M / rh**3
    else:
        nn = (l - 1) * (l + 2) / 2.0
        q = (2 * nn**2 * (nn + 1) * rh**3 + 6 * nn**2 * M * rh**2 + 18 * nn * M**2 * rh + 18 * M**3) \
            / (rh**3 * (nn * rh + 3 * M)**2)
    a1 = q / (1.0 / (2 * M) - 2j * w)
    ph = np.exp(-1j * w * rs)
    Psi = ph * (1 + a1 * x)
    dPsi = ph * (-1j * w / (1 - 2 * M / r) * (1 + a1 * x) + a1)
    return r, np.array([Psi, dPsi])


def bh_reflectivity(w, l, parity, M=1.0, rfar=None):
    """|B_out/B_in|^2 and transmissivity 1-|R|^2 check for a Schwarzschild BH (real w)."""
    r0, y0 = horizon_start(w, l, parity, M)
    if rfar is None:
        rfar = max(120.0 * M, 60.0 / w)
    sol = integrate_real(y0, r0, rfar, l, parity, w, M)
    P, dP = sol.y[0, -1], sol.y[1, -1]
    Pu, dPu, _ = up_asymptotic(rfar, l, parity, w, M)
    Pi, dPi = np.conj(Pu), np.conj(dPu)
    A = np.array([[Pi, Pu], [dPi, dPu]])
    Bin, Bout = np.linalg.solve(A, [P, dP])
    return Bout / Bin, 1.0 / Bin      # reflection amplitude, transmission amplitude (horizon)
