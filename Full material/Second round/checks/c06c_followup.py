"""CHECK 06c -- (1) model-A row choices judged with the first round's own (global) residual
normalisation; (2) even-C vs odd reflectivity at high frequency (does even 'merge' with odd?)."""
import sys, os, numpy as np
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fr_code_pristine'))
from shellgw import junction as J
import shellgw.junction_even_A as EA
import ind_solver as I

def gres(A, B, X, rows):
    out = 0
    for k in range(2):
        x, y = X[:, k], np.eye(2)[:, k]
        num = np.abs(A[rows] @ x - B[rows] @ y).max()
        den = (np.abs(A) @ np.abs(x) + np.abs(B) @ np.abs(y)).max()
        out = max(out, num/den)
    return out

print('(1) model A, R=3, l=2: global-normalised residual of the 2 NON-imposed rows, for each choice')
for w in (1.0, 10.0, 100.0):
    A = EA.A(w, 6, 3.0, 1.0, 0.0); B = EA.B(w, 6, 3.0, 1.0, 0.0)
    X0 = np.linalg.solve(A[J.EVOLUTION_ROWS], B[J.EVOLUTION_ROWS])
    res = []
    for rows in combinations(range(8), 6):
        rows = list(rows)
        if np.linalg.cond(A[rows]) > 1e12: continue
        X = np.linalg.solve(A[rows], B[rows])
        other = [i for i in range(8) if i not in rows]
        res.append((gres(A, B, X, other), rows, np.abs(X[-2:] - X0[-2:]).max()/np.abs(X0[-2:]).max()))
    res.sort()
    print('  w=%5.0f:' % w)
    for r_ in res[:5]:
        print('     rows %s  residual of dropped rows %.2e   |T - T_first|/|T_first| = %.2e' % (r_[1], r_[0], r_[2]))
    print('     ... worst: rows %s residual %.2e' % (res[-1][1], res[-1][0]))

print('(2) R_even-C / R_odd at high frequency, l=2, vs (Delta_even/Delta_odd)^2 from the matrix')
for R in (3.0, 6.0, 10.0):
    s = np.sqrt(1 - 2/R); k = (1 - s)/(4*s); vs2 = 2*k/(1 + k)
    for w in (2.0, 5.0, 10.0):
        Re, _ = I.RT(w, 2, 'even', R, 'C'); Ro, _ = I.RT(w, 2, 'odd', R)
        T = J.T_even(w, 2, R, 1.0, 'C', vs2)
        ratio = (T[1, 0].real/J.T_odd(R)[1, 0])**2
        print('  R=%4.1f w=%4.1f  R_even=%.4e R_odd=%.4e  ratio=%.4f   (Delta_e/Delta_o)^2=%.4f' % (R, w, Re, Ro, Re/Ro, ratio))
