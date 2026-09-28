"""
Linearised junction conditions at the shell, as 2x2 transfer matrices
    (Psi_+, dPsi_+/dr)|_{R+} = T(omega) (Psi_-, dPsi_-/dr)|_{R-},
with Psi_+ the exterior master function (CPM for odd, Zerilli-Moncrief for even,
MP05 normalisations) and Psi_- the interior one (same definitions with M = 0,
interior Minkowski time T = sqrt(f(R)) t).  Derivation: derivation document
Sec. 4; symbolic, first-principles: code/symbolic/derive_junction.py.

ODD PARITY (both models C and A, and any perfect-fluid shell, omega != 0):
    Psi_+ = Psi_-                                                        (J1)
    sqrt(f) dPsi_+/dr - dPsi_-/dr = (1 - sqrt f)/R Psi = 4 pi sigma Psi  (J2)
  i.e. with proper radial distance l (dl = dr/sqrt f outside, dl = dr inside)
    [dPsi/dl] = 4 pi sigma Psi,
  and with the global tortoise coordinate x (x = r* outside, x = r/sqrt f(R) inside)
    [dPsi/dx] = Delta_odd Psi,   Delta_odd = sqrt(f)(1 - sqrt f)/R.
EVEN PARITY: full 2x2 matrix generated symbolically (shell displacement and
shell-matter motion eliminated), loaded from code/symbolic/junction_even_*.py.
"""
import numpy as np
from importlib import import_module


def T_odd(R, M=1.0):
    """Odd-parity transfer matrix (J1)-(J2); frequency independent."""
    s = np.sqrt(1.0 - 2.0 * M / R)
    return np.array([[1.0, 0.0], [(1.0 - s) / (R * s), 1.0 / s]])


def Delta_odd(R, M=1.0):
    """Jump coefficient in [dPsi/dx] = Delta Psi (global tortoise coordinate)."""
    s = np.sqrt(1.0 - 2.0 * M / R)
    return s * (1.0 - s) / R


_even_cache = {}

# row order of the generated systems: [h_tt], [h_tA], [h_tr], [h_tf],
#                                     Israel_tt, Israel_tA, Israel_tr, Israel_tf
EVOLUTION_ROWS = [0, 1, 2, 3, 6, 7]
CONSTRAINT_ROWS = [4, 5]


def _mod(model):
    if model not in _even_cache:
        _even_cache[model] = import_module('shellgw.junction_even_%s' % model)
    return _even_cache[model]


_mp_cache = {}
MP_BELOW = 2.3      # use 40-digit arithmetic for R < MP_BELOW * M (cancellations ~ (R-2M)^-k)


def _mp_AB(model):
    """High-precision (mpmath) versions of A, B lambdified from the saved sympy system."""
    if model not in _mp_cache:
        import os
        import pickle
        import sympy as sp
        path = os.path.join(os.path.dirname(__file__), '..', 'symbolic', 'junction_even_%s.pkl' % model)
        with open(path, 'rb') as fh:
            info = pickle.load(fh)
        syms = sorted(set().union(*[info['A'].free_symbols, info['B'].free_symbols]), key=str)
        names = {str(s): s for s in syms}
        order = [names[k] for k in ('omega', 'lambda', 'R', 'M') if k in names]
        order += [names[k] for k in ('v_s2',) if k in names]
        fA = sp.lambdify(order, info['A'], 'mpmath')
        fB = sp.lambdify(order, info['B'], 'mpmath')
        _mp_cache[model] = (fA, fB, [str(s) for s in order])
    return _mp_cache[model]


def _AB(model, w, lam, R, M, vs2):
    if R < MP_BELOW * M:
        import mpmath as mp
        fA, fB, order = _mp_AB(model)
        with mp.workdps(40):
            vals = dict(omega=mp.mpc(w), **{'lambda': mp.mpf(lam)}, R=mp.mpf(R), M=mp.mpf(M), v_s2=mp.mpf(vs2))
            args = [vals[k] for k in order]
            A = fA(*args)
            B = fB(*args)
            return A, B, True
    mod = _mod(model)
    return mod.A(w, lam, R, M, vs2), mod.B(w, lam, R, M, vs2), False


def _solve(A, B, rows, is_mp):
    if is_mp:
        import mpmath as mp
        with mp.workdps(40):
            if rows is not None:
                A = mp.matrix([[A[i, j] for j in range(A.cols)] for i in rows])
                B = mp.matrix([[B[i, j] for j in range(B.cols)] for i in rows])
            cols = []
            for j in range(B.cols):
                x = mp.lu_solve(A, mp.matrix([B[i, j] for i in range(B.rows)]))
                cols.append([complex(x[i]) for i in range(A.cols)])
            return np.array(cols).T
    if rows is not None:
        A, B = A[rows], B[rows]
    return np.linalg.solve(A, B)


def T_even(w, l, R, M=1.0, model='C', vs2=0.0):
    """Even-parity transfer matrix from the symbolic derivation.

    Model C (static fluid shell): 8 equations, 8 unknowns (xi_+, xi_-, zeta, eta,
      delta sigma, U, Psi_+, Psi_+'), solved exactly.
    Model A (frozen domain wall): 8 equations, 6 unknowns. The frozen-coefficient
      ansatz is not an exact solution of the moving-shell problem, so the tau-tau and
      tau-A Israel equations (which for an exact solution follow from the Codazzi
      identity) are not identically satisfied. We solve the six evolution equations
      ([h_ab] = 0 and the angular Israel equations) and monitor the two constraint
      rows with constraint_residual()."""
    lam = l * (l + 1)
    A, B, is_mp = _AB(model, w, lam, R, M, vs2)
    X = _solve(A, B, EVOLUTION_ROWS if model == 'A' else None, is_mp)
    return X[-2:, :]


def constraint_residual(w, l, R, M=1.0, model='A', vs2=0.0):
    """Relative violation of the tau-tau and tau-A Israel constraints by the model-A
    solution (0 for model C): max over the two interior basis vectors of
    |A_c x - B_c y| / (|A_c| |x| + |B_c| |y|)."""
    mod = _mod(model)
    lam = l * (l + 1)
    A = mod.A(w, lam, R, M, vs2)
    B = mod.B(w, lam, R, M, vs2)
    if model == 'A':
        X = np.linalg.solve(A[EVOLUTION_ROWS], B[EVOLUTION_ROWS])
    else:
        X = np.linalg.solve(A, B)
    Ac, Bc = A[CONSTRAINT_ROWS], B[CONSTRAINT_ROWS]
    out = 0.0
    for k in range(2):
        x = X[:, k]
        y = np.eye(2)[:, k]
        num = np.abs(Ac @ x - Bc @ y).max()
        den = (np.abs(Ac) @ np.abs(x) + np.abs(Bc) @ np.abs(y)).max()
        out = max(out, num / den)
    return out


def transfer(parity, w, l, R, M=1.0, model='C', vs2=0.0):
    if parity == 'odd':
        return T_odd(R, M).astype(complex)
    return T_even(w, l, R, M, model, vs2)
