"""CHECK 09 -- (a) poles of T_even (zeros of det A) vs matter QNMs; (b) matter modes missing from the catalogue."""
import sys, os, numpy as np, mpmath as mp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fr_code_pristine'))
import shellgw.junction_even_C as EC
from shellgw.studies import pn_matter_seeds
import ind_solver as I
for R in (6.0, 10.0):
    s = np.sqrt(1 - 2/R); k = (1 - s)/(4*s); vs2 = 2*k/(1 + k); sc = R**-1.5
    f = lambda w: complex(np.linalg.det(EC.A(complex(w), 6, R, 1.0, vs2)))
    roots = set()
    for g in (0.5*sc, 1.0*sc, 1.5*sc, 2*sc, 0.5j*sc, 1j*sc):
        try:
            z = complex(mp.findroot(lambda x: mp.mpc(f(complex(x))), mp.mpc(g)))
            roots.add((round(z.real/sc, 4), round(z.imag/sc, 4)))
        except Exception:
            pass
    print('(a) R=%g: poles of T_even in units sqrt(M/R^3):' % R, sorted(roots))
print('    matter QNMs in the same units: R=6: 1.3147-0.0008i, 0.590i ; R=10: 1.398, 0.557i')
print('(b) stable matter pair, l=2, continued from the first-round track (R=2.529M: 0.212872-0.000221i):')
w = 0.21287159643569079 - 0.00022056236005921142j
for R in (2.5, 2.4, 2.3, 2.2):
    w = I.qnm(w, 2, 'even', R, 'C'); print('    R=%.2f  %.6f%+.6fi' % (R, w.real, w.imag))
w = I.qnm(0.19681264130029782 - 0.0003099470424562935j, 2, 'even', 3.0, 'C'); print('    R=3.00  %.6f%+.6fi' % (w.real, w.imag))
for l in (2, 3, 4):
    sR, sI = pn_matter_seeds(l, 2.0); R = 2.2; sc = R**-1.5; found = None
    for fac in (1.0, 0.8, 0.6, 1.3, 0.45, 1.6):
        try:
            z = I.qnm(1j*abs(sI)*sc*fac, l, 'even', R, 'C')
            if abs(z.real) < 1e-6 and z.imag > 0:
                found = z; break
        except Exception:
            pass
    print('    unstable matter mode l=%d, R=2.2M:' % l, found)
