"""
CHECK 06 -- properties of the first-round even-parity transfer matrices (the object under test),
evaluated from an untouched copy of the first-round code (fr_code_pristine).

 (a) det T^C * sqrt(f) = 1 ?   (claimed exact) -- random real & complex w, l, R, v_s^2
 (b) is T^C real for real w?  flux conservation  J = Im(Psi^* dPsi/dx) for arbitrary interior data
 (c) transparent limit M -> 0
 (d) double vs 40-digit precision near R = 2M (claimed 1e-6 at R = 2.2 in double)
 (e) model A: dependence of T on WHICH 6 of the 8 equations are imposed
 (f) model A: constraint residual ~ 1/w ; |R+T-1| ~ w^-2 ?  (from first-round scattering.json)
 (g) high-frequency behaviour: T^C_even vs T_odd; Psi discontinuous in even parity
"""
import sys
import os
import json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fr_code_pristine'))
from shellgw import junction as J            # noqa: E402
import shellgw.junction_even_C as EC         # noqa: E402
import shellgw.junction_even_A as EA         # noqa: E402

rng = np.random.default_rng(7)
out = {}


def sq(R):
    return np.sqrt(1 - 2/R)


# (a) + (b)
worst_det, worst_imag, worst_flux = 0, 0, 0
worst_det_c = 0
for _ in range(300):
    l = int(rng.integers(2, 6))
    R = float(rng.uniform(2.35, 30))
    vs2 = float(rng.uniform(0.01, 0.99))
    w = float(rng.uniform(0.01, 3))
    T = J.T_even(w, l, R, 1.0, 'C', vs2)
    worst_det = max(worst_det, abs(np.linalg.det(T)*sq(R) - 1))
    worst_imag = max(worst_imag, np.abs(T.imag).max()/np.abs(T).max())
    # flux: interior data y = (Psi, dPsi/dr); J_int = sqrt f Im(conj(P) dP), J_ext = f Im(conj(P+) dP+)
    y = rng.normal(size=2) + 1j*rng.normal(size=2)
    yp = T @ y
    Jin = sq(R)*np.imag(np.conj(y[0])*y[1])
    Jout = sq(R)**2*np.imag(np.conj(yp[0])*yp[1])
    worst_flux = max(worst_flux, abs(Jout - Jin)/np.abs(y).max()**2)
    wc = w - 1j*float(rng.uniform(0, 1))
    Tc = J.T_even(wc, l, R, 1.0, 'C', vs2)
    worst_det_c = max(worst_det_c, abs(np.linalg.det(Tc)*sq(R) - 1))
print('(a) max |det T^C sqrt f - 1| : real w %.1e, complex w %.1e  (300 random samples, R in [2.35,30], l=2..5)'
      % (worst_det, worst_det_c))
print('(b) max |Im T^C|/|T^C| for real w: %.1e ;  max flux mismatch %.1e' % (worst_imag, worst_flux))
out.update(det_real=worst_det, det_complex=worst_det_c, imagT=worst_imag, flux=worst_flux)

# (c)
for Msmall in (1e-4, 1e-6, 1e-8):
    tc = np.abs(J.T_even(0.5, 2, 5.0, Msmall, 'C', 0.3) - np.eye(2)).max()
    ta = np.abs(J.T_even(0.5, 2, 5.0, Msmall, 'A') - np.eye(2)).max()
    print('(c) M=%.0e: |T^C - 1| = %.1e, |T^A - 1| = %.1e' % (Msmall, tc, ta))
    out['transparent_%g' % Msmall] = [tc, ta]

# (d) precision near 2M: double-precision modules directly vs mp path
print('(d) det T^C sqrt f - 1 : double vs 40-digit')
for R in (2.3, 2.2, 2.1, 2.05, 2.02, 2.01):
    vs2 = 2*((1 - sq(R))/(4*sq(R)))/(1 + (1 - sq(R))/(4*sq(R)))
    A = EC.A(0.4, 6, R, 1.0, vs2)
    B = EC.B(0.4, 6, R, 1.0, vs2)
    Td = np.linalg.solve(A, B)[-2:]
    Tm = J.T_even(0.4, 2, R, 1.0, 'C', vs2)
    print('    R=%.2f  double: %.1e   mp: %.1e   cond(A)=%.1e' % (R, abs(np.linalg.det(Td)*sq(R) - 1),
                                                                abs(np.linalg.det(Tm)*sq(R) - 1), np.linalg.cond(A)))

# (e) model A row choice
from itertools import combinations  # noqa: E402
print('(e) model A: spread of T over all C(8,6)=28 choices of imposed rows (w, R=3, l=2)')
for w in (0.5, 2.0, 10.0, 50.0):
    A = EA.A(w, 6, 3.0, 1.0, 0.0)
    B = EA.B(w, 6, 3.0, 1.0, 0.0)
    Ts = []
    for rows in combinations(range(8), 6):
        rows = list(rows)
        try:
            if np.linalg.cond(A[rows]) > 1e12:
                continue
            Ts.append(np.linalg.solve(A[rows], B[rows])[-2:])
        except np.linalg.LinAlgError:
            pass
    T0 = np.linalg.solve(A[J.EVOLUTION_ROWS], B[J.EVOLUTION_ROWS])[-2:]
    Tls = np.linalg.lstsq(A, B, rcond=None)[0][-2:]
    spread = max(np.abs(T - T0).max() for T in Ts)/np.abs(T0).max()
    print('    w=%5.1f  %d well-posed choices; max rel. spread vs first-round choice %.2e ; lstsq vs choice %.2e'
          % (w, len(Ts), spread, np.abs(Tls - T0).max()/np.abs(T0).max()))
    out['A_rowchoice_w%g' % w] = spread

# (f) scaling of model-A violations from first-round data
S = json.load(open('../../First round/results/scattering.json'))
for R in (3.0, 6.0, 10.0):
    c = [c for c in S['curves'] if c['parity'] == 'even' and c['model'] == 'A' and c['l'] == 2 and c['R'] == R][0]
    w = np.array(c['w'])
    e = np.abs(np.array(c['R1'], float) + np.array(c['T1'], float) - 1)
    for a, b in ((0.5, 1.0), (1.0, 2.0)):
        i, k = np.argmin(abs(w - a)), np.argmin(abs(w - b))
        print('(f) model A R=%g: local exponent of |R+T-1| between wM=%.1f and %.1f: %.2f' % (R, a, b, np.log(e[k]/e[i])/np.log(w[k]/w[i])))
        out['A_exponent_R%g_%g_%g' % (R, a, b)] = float(np.log(e[k]/e[i])/np.log(w[k]/w[i]))
# extend to higher frequency with the first-round method-1 code itself
from shellgw.scattering import barrier_RT  # noqa: E402
print('(f) model A, R=3, extended frequency range (first-round Method 1):')
prev = None
for w in (2.0, 4.0, 8.0, 16.0, 32.0):
    Rc, Tc = barrier_RT(w, 2, 'even', 3.0, 1.0, 'A', 0.0, 1)
    e = abs(Rc + Tc - 1)
    cr = J.constraint_residual(w, 2, 3.0, 1.0, 'A')
    txt = '    w=%5.1f  |R+T-1| = %.3e   R = %.3e   constraint = %.2e' % (w, e, Rc, cr)
    if prev:
        txt += '   exponent %.2f' % (np.log(e/prev[1])/np.log(w/prev[0]))
    print(txt)
    prev = (w, e)
    out.setdefault('A_highw', []).append(dict(w=w, viol=e, R=Rc, constr=cr))

# (g)
print('(g) high frequency: |T^C_even - T_odd| and T^C_even[0,:] (Psi continuity) at R=6, l=2')
for w in (0.5, 2, 10, 50, 200):
    T = J.T_even(w, 2, 6.0, 1.0, 'C', 2*0.0562/(1.0562))
    print('    w=%6.1f  |T_even - T_odd| = %.3e   T_even row 0 = [%.4f, %.4f]' % (w, np.abs(T - J.T_odd(6.0)).max(),
                                                                                 T[0, 0].real, T[0, 1].real))
with open('out_c06.json', 'w') as fh:
    json.dump(out, fh, indent=1, default=float)
