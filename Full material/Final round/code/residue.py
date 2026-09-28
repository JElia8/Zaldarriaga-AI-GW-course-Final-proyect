"""Residue of the closed-system reflection amplitude S_even(w) at the unstable matter mode w = i*gamma.

S = B_out/B_in, with the exterior solution (regular interior carried through the even junction) projected at a
moderate radius r0 onto the normalised asymptotic solutions  Psi_up ~ e^{+i w r*},  Psi_in(w) = Psi_up(-w) ~ e^{-i w r*}.
For Im w > 0 the outgoing solution decays with r, so r0 must be moderate (B_out/B_in ~ e^{-2 gamma r*} at r0)."""
import sys, os
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'Third round', 'code', 'fr_code'))
import warnings
warnings.filterwarnings('ignore')
from shellgw.scattering import exterior_data
from shellgw.exterior import integrate_real, up_asymptotic
from parity_study import vs2


def B_coeffs(w, R, r0, l=2, par='even'):
    yp = exterior_data(w, l, par, R, 1.0, 'C', vs2(R) if par == 'even' else 0.0, 'regular')
    sol = integrate_real(yp, R, r0, l, par, w, 1.0, rtol=1e-12, atol=1e-14)
    P, dP = sol.y[0, -1], sol.y[1, -1]
    Pu, dPu, eu = up_asymptotic(r0, l, par, w)
    Pi, dPi, ei = up_asymptotic(r0, l, par, -w)
    Bin, Bout = np.linalg.solve(np.array([[Pi, Pu], [dPi, dPu]]), [P, dP])
    return Bin, Bout, max(abs(eu), abs(ei))


def residue(R, gamma0, r0):
    from scipy.optimize import root
    f = lambda x: (lambda b: [b.real, b.imag])(B_coeffs(x[0] + 1j * x[1], R, r0)[0] / B_coeffs(x[0] + 1j * x[1], R, r0)[1])
    # zero of B_in (normalised by B_out, which is regular there)
    sol = root(f, [0.0, gamma0], tol=1e-13)
    wp = sol.x[0] + 1j * sol.x[1]
    h = 1e-5
    Bp, Bo, _ = B_coeffs(wp + h, R, r0)
    Bm, _, _ = B_coeffs(wp - h, R, r0)
    dBin = (Bp - Bm) / (2 * h)
    _, Bout, err = B_coeffs(wp, R, r0)
    return wp, Bout / dBin, err


if __name__ == '__main__':
    from parity_study import S_shell
    # 1) validation on the real axis against the standard closed-system S
    for R, r0 in ((3.0, 40.0), (2.2, 30.0), (6.0, 90.0)):
        for w in (0.2, 0.45):
            Bin, Bout, err = B_coeffs(w, R, r0)
            print('R=%g w=%.2f  S(r0=%g) = %s   S(standard) = %s' % (R, w, r0, np.round(Bout / Bin, 8), np.round(S_shell(w, 2, 'even', R), 8)))
    # 2) residues at the unstable pole, stability of the result against r0
    for R, g, r0s in ((3.0, 0.13321, (30.0, 40.0, 50.0)), (2.2, 0.24429, (22.0, 30.0, 36.0)), (6.0, 0.040138, (70.0, 90.0, 120.0))):
        for r0 in r0s:
            wp, res, err = residue(R, g, r0)
            print('R=%g r0=%g: pole %s  residue %s  (series err %.1e)' % (R, r0, np.round(wp, 7), np.round(res, 6), err))
