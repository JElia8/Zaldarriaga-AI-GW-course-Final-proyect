"""CHECK 06b -- follow-ups: (1) normalized residuals of every model-A row choice;
(2) high-frequency limit of T_even^C vs T_odd element by element, and implied delta strength."""
import sys, os, numpy as np
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fr_code_pristine'))
from shellgw import junction as J
import shellgw.junction_even_A as EA

def nres(A, B, X):
    """max over rows and over the 2 interior basis vectors of |A_i x - B_i y|/(|A_i||x| + |B_i||y|)"""
    out = 0
    for k in range(2):
        x, y = X[:, k], np.eye(2)[:, k]
        num = np.abs(A @ x - B @ y)
        den = np.abs(A) @ np.abs(x) + np.abs(B) @ np.abs(y)
        out = max(out, (num/den).max())
    return out

print('(1) model A, R=3, l=2: normalized max residual over ALL 8 rows, per choice of 6 imposed rows')
for w in (0.5, 2.0, 10.0, 50.0):
    A = EA.A(w, 6, 3.0, 1.0, 0.0); B = EA.B(w, 6, 3.0, 1.0, 0.0)
    res = []
    for rows in combinations(range(8), 6):
        rows = list(rows)
        if np.linalg.cond(A[rows]) > 1e12: continue
        X = np.linalg.solve(A[rows], B[rows])
        res.append((nres(A, B, X), rows))
    res.sort()
    X0 = np.linalg.solve(A[J.EVOLUTION_ROWS], B[J.EVOLUTION_ROWS])
    Xl = np.linalg.lstsq(A, B, rcond=None)[0]
    # row-normalised least squares
    sc = 1/np.sqrt((np.abs(A)**2).sum(1) + (np.abs(B)**2).sum(1))
    Xn = np.linalg.lstsq(A*sc[:, None], B*sc[:, None], rcond=None)[0]
    print('  w=%5.1f first-round rows %s: %.2e | best choice %s: %.2e | 2nd best %.2e | worst %.2e | lstsq %.2e | row-normalised lstsq %.2e, |T_n - T_first|/|T| = %.2e'
          % (w, J.EVOLUTION_ROWS, nres(A, B, X0), res[0][1], res[0][0], res[1][0], res[-1][0], nres(A, B, Xl), nres(A, B, Xn),
             np.abs(Xn[-2:] - X0[-2:]).max()/np.abs(X0[-2:]).max()))
print('(2) T_even^C - T_odd element by element (l=2)')
for R in (3.0, 6.0, 10.0):
    s = np.sqrt(1 - 2/R); k = (1 - s)/(4*s); vs2 = 2*k/(1 + k)
    for w in (10.0, 100.0, 1000.0):
        T = J.T_even(w, 2, R, 1.0, 'C', vs2); To = J.T_odd(R)
        d = T - To
        # implied delta strength in the global tortoise coordinate: [dPsi/dx] = f T21 Psi when T11=1, T22 = 1/s
        print('  R=%4.1f w=%6.0f  dT = [[%.2e, %.2e], [%.3e, %.2e]]   Delta_even/Delta_odd = %.4f'
              % (R, w, d[0,0].real, d[0,1].real, d[1,0].real, d[1,1].real, T[1,0].real/To[1,0]))
