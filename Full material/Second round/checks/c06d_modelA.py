"""CHECK 06d -- model A: keep the 4 metric-continuity rows (first junction condition, must hold
exactly) and impose 2 of the 4 Israel rows; compare T and the reflectivity."""
import sys, os, numpy as np
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fr_code_pristine'))
from shellgw import junction as J
import shellgw.junction_even_A as EA
from shellgw.interior import ingoing_at_shell
from shellgw.exterior import up_at_R_cf
names = {4: 'tt', 5: 'tA', 6: 'trace', 7: 'tracefree'}

def RT_from_T(T, w, l, R):
    yp = T @ ingoing_at_shell(w, l, R)
    up = up_at_R_cf(w, l, 'even', R)
    f = 1 - 2/R
    L = f*up[1]/up[0]
    up = up*np.sqrt(w/L.imag)/abs(up[0])
    Bin, Bout = np.linalg.solve(np.array([[np.conj(up[0]), up[0]], [np.conj(up[1]), up[1]]]), yp)
    return abs(Bout/Bin)**2, 1/abs(Bin)**2

def gres(A, B, X, rows):
    out = 0
    for k in range(2):
        x, y = X[:, k], np.eye(2)[:, k]
        out = max(out, np.abs(A[rows] @ x - B[rows] @ y).max()/(np.abs(A) @ np.abs(x) + np.abs(B) @ np.abs(y)).max())
    return out

for R in (3.0, 6.0):
    print('R = %g' % R)
    for w in (0.5, 1.0, 3.0, 10.0, 30.0):
        A = EA.A(w, 6, R, 1.0, 0.0); B = EA.B(w, 6, R, 1.0, 0.0)
        X0 = np.linalg.solve(A[J.EVOLUTION_ROWS], B[J.EVOLUTION_ROWS])
        line = '  w=%5.1f ' % w
        for pair in combinations((4, 5, 6, 7), 2):
            rows = [0, 1, 2, 3] + list(pair)
            if np.linalg.cond(A[rows]) > 1e13:
                line += ' | %s+%s: singular' % (names[pair[0]], names[pair[1]]); continue
            X = np.linalg.solve(A[rows], B[rows])
            other = [i for i in (4, 5, 6, 7) if i not in pair]
            Rc, Tc = RT_from_T(X[-2:], w, 2, R)
            line += ' | %s+%s: res %.1e dT %.1e R %.3e R+T-1 %.1e' % (names[pair[0]], names[pair[1]], gres(A, B, X, other),
                     np.abs(X[-2:] - X0[-2:]).max()/np.abs(X0[-2:]).max(), Rc, Rc + Tc - 1)
        print(line)
