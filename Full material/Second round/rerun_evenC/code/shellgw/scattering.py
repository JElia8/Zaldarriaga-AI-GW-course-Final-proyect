"""
Scattering of gravitational waves off the shell (real omega).

Definitions (derivation document Sec. 5.1):

 (i) Barrier-plus-shell coefficients.  The exterior potential and the shell are
     treated as a scatterer between spatial infinity and the flat interior; the
     interior is "open", containing only the ingoing Riccati-Hankel wave
         Psi_- = hin_l(nu r),  nu = omega/sqrt(f(R)).
     Exterior:  Psi_+ = B_in e^{-i w r*} + B_out e^{+i w r*}  (r* -> inf).
         Rcoef(w) = |B_out/B_in|^2,   Tcoef(w) = 1/|B_in|^2.
     The conserved Wronskian flux J = Im(Psi* dPsi/dx) (x global tortoise
     coordinate; J_int = -w for hin, J_ext = w(|B_out|^2 - |B_in|^2)) implies
     Rcoef + Tcoef = 1 when the junction conserves flux: this is a genuine test
     of the junction conditions and of the numerics.
     Limits: omega -> 0: Rcoef -> 1 (tidal/static response of the shell);
     R -> 2M: Rcoef -> Schwarzschild BH reflectivity (Vishveshwara 1970);
     M -> 0: Rcoef -> 0 (flat space, no scatterer).

 (ii) Closed (physical) system: interior regular, Psi_- = jhat_l(nu r).
     No absorption anywhere, so S(w) = B_out/B_in has |S| = 1 exactly;
     S = e^{2 i delta(w)}; resonances (QNMs) show up as rapid 2 delta jumps.

Two independent methods:
  Method 1 (transfer matrix): exterior in/up basis at R from the continued
     fraction (exterior.up_at_R_cf); |B| normalisation from the Wronskian.
  Method 2 (shooting): DOP853 integration from R to r_far, projection on the
     asymptotic in/up solutions (high-order asymptotic series), plus the
     Boyanov et al. (2024, App. B1) estimator |i w Psi + Psi'|/|i w Psi - Psi'|
     averaged over several periods.
"""
import numpy as np
from .interior import ingoing_at_shell, regular_at_shell, regular_numerical
from .junction import transfer
from .exterior import up_at_R_cf, integrate_real, up_asymptotic


def _shell_vector(w, l, R, M, interior):
    if interior == 'ingoing':
        return ingoing_at_shell(w, l, R, M)
    if interior == 'regular':
        return regular_at_shell(w, l, R, M)
    if interior == 'regular_num':
        return regular_numerical(w, l, R, M)
    raise ValueError(interior)


def exterior_data(w, l, parity, R, M=1.0, model='C', vs2=0.0, interior='ingoing'):
    """(Psi_+, dPsi_+/dr) at R+ from the interior solution through the junction."""
    ym = _shell_vector(w, l, R, M, interior)
    return transfer(parity, w, l, R, M, model, vs2) @ ym


def coeffs_method1(w, l, parity, R, M=1.0, model='C', vs2=0.0, interior='ingoing'):
    """B_in, B_out by the transfer-matrix method (continued-fraction exterior basis).
    Only |B_in|, |B_out| are meaningful (the up-solution phase at R is not fixed)."""
    yp = exterior_data(w, l, parity, R, M, model, vs2, interior)
    up = up_at_R_cf(w, l, parity, R, M)
    f = 1.0 - 2.0 * M / R
    Lstar = f * up[1] / up[0]                      # dPsi/dr* / Psi of up solution
    # |A|^2 of up = e^{i w r*}-normalised solution:  |Psi_up(R)|^2 = w / Im(L*)
    scale = np.sqrt(w / Lstar.imag) / abs(up[0])
    up = up * scale
    inn = np.conj(up)
    A = np.array([[inn[0], up[0]], [inn[1], up[1]]])
    Bin, Bout = np.linalg.solve(A, yp)
    return Bin, Bout


def coeffs_method2(w, l, parity, R, M=1.0, model='C', vs2=0.0, interior='ingoing',
                   rfar=None, window=True):
    """B_in, B_out by direct DOP853 integration to r_far and projection onto the
    asymptotic basis; optionally the Boyanov et al. windowed ratio estimate."""
    yp = exterior_data(w, l, parity, R, M, model, vs2, interior)
    if rfar is None:
        rfar = max(150.0 * M, 60.0 / w)
    period = 2 * np.pi / w
    rend = rfar + (4 * period if window else 0.0)
    sol = integrate_real(yp, R, rend, l, parity, w, M, dense=True)
    P, dP = sol.sol(rfar)
    up = up_asymptotic(rfar, l, parity, w, M)
    Pu, dPu = up[0], up[1]
    A = np.array([[np.conj(Pu), Pu], [np.conj(dPu), dPu]])
    Bin, Bout = np.linalg.solve(A, [P, dP])
    est = None
    if window:
        rr = np.linspace(rfar, rend, 400)
        Y = sol.sol(rr)
        f = 1.0 - 2.0 * M / rr
        Ls = f * Y[1] / Y[0]                        # dPsi/dr* / Psi
        est = np.mean(np.abs(1j * w + Ls) / np.abs(1j * w - Ls))
    return Bin, Bout, est


def barrier_RT(w, l, parity, R, M=1.0, model='C', vs2=0.0, method=1):
    """Rcoef, Tcoef of the barrier-plus-shell problem (definition (i))."""
    if method == 1:
        Bin, Bout = coeffs_method1(w, l, parity, R, M, model, vs2, 'ingoing')
    else:
        Bin, Bout, _ = coeffs_method2(w, l, parity, R, M, model, vs2, 'ingoing', window=False)
    return abs(Bout / Bin)**2, 1.0 / abs(Bin)**2


def closed_S(w, l, parity, R, M=1.0, model='C', vs2=0.0):
    """S = B_out/B_in of the physical (regular interior) system, method 2 (phase meaningful)."""
    Bin, Bout, _ = coeffs_method2(w, l, parity, R, M, model, vs2, 'regular', window=False)
    return Bout / Bin
