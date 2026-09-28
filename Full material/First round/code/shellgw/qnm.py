"""
Quasinormal modes of the shell spacetime: complex w with Psi regular at r = 0,
junction conditions at r = R, and purely outgoing waves at infinity.

QNM condition: Wronskian at r = R+ between the solution coming from the regular
interior through the junction, y_J = T(w) (jhat, jhat'), and the exterior purely
outgoing solution y_up:
        W(w) = Psi_J dPsi_up/dr - dPsi_J/dr Psi_up = 0.

Method 1: y_up from the continued fraction (Leaver-type, exterior.up_at_R_cf).
Method 2: y_up from the high-order asymptotic series at a far point on the complex
          ray r = R + s e^{i phi}, phi = pi/2 - arg(w), along which the outgoing
          solution is exponentially *subdominant* at large s, so that inward DOP853
          integration back to R is stable (standard complex-scaling idea; cf. the
          Boyanov et al. 2024 App. B1 strategy of integrating inward from a
          high-order exterior expansion to a matching point).
Roots: scipy.optimize.root on (Re W, Im W).
"""
import numpy as np
from scipy.optimize import root
from .interior import regular_at_shell
from .junction import transfer
from .exterior import up_at_R_cf, up_asymptotic, integrate_ray


def y_junction(w, l, parity, R, M=1.0, model='C', vs2=0.0):
    return transfer(parity, w, l, R, M, model, vs2) @ regular_at_shell(w, l, R, M)


def up_ray(w, l, parity, R, M=1.0, L=None):
    """Outgoing solution at R via asymptotic series on the complex ray + inward DOP853."""
    # direction of fastest decay of e^{i w r}, clamped so that the ray never passes
    # near r = 0 or r = 2M (needed for purely imaginary, damped frequencies)
    phi = min(np.pi / 2 - np.angle(w), 2.0)
    if L is None:
        L = max(60.0 * M, 40.0 / abs(w))
    rfar = R + L * np.exp(1j * phi)
    P, dP, err = up_asymptotic(rfar, l, parity, w, M)
    # normalise: along the ray |Psi_up| ~ e^{-|w| s} is tiny at the far end, and an
    # absolute tolerance would then be meaningless (the Wronskian is scale-free)
    nrm = abs(P) + abs(dP)
    y = integrate_ray([P / nrm, dP / nrm], rfar, R + 0j, l, parity, w, M, atol=1e-30)
    return y


def wronskian(w, l, parity, R, M=1.0, model='C', vs2=0.0, method=1):
    """Normalised Wronskian W / (|y_J| |y_up|); returns a large value outside the
    physical domain (Re w <= 0 or Im w > 0) so that root finders are pushed back."""
    if not (np.isfinite(w) and w.real > -1e-3 and abs(w) > 1e-7 and w.imag < 5.0):
        return 1e3 + 0j
    try:
        yJ = y_junction(w, l, parity, R, M, model, vs2)
        yu = up_at_R_cf(w, l, parity, R, M) if method == 1 else up_ray(w, l, parity, R, M)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return 1e3 + 0j
    W = yJ[0] * yu[1] - yJ[1] * yu[0]
    v = W / (np.linalg.norm(yJ) * np.linalg.norm(yu))
    return v if np.isfinite(v) else 1e3 + 0j


def find_qnm(guess, l, parity, R, M=1.0, model='C', vs2=0.0, method=1, tol=1e-13):
    def F(x):
        v = wronskian(x[0] + 1j * x[1], l, parity, R, M, model, vs2, method)
        return [v.real, v.imag]
    sol = root(F, [guess.real, guess.imag], method='hybr', tol=tol)
    w = sol.x[0] + 1j * sol.x[1]
    res = abs(wronskian(w, l, parity, R, M, model, vs2, method))
    return w, res, sol.success


def scan_fast(l, parity, R, M=1.0, model='C', vs2=0.0, re=(0.02, 2.0), im=(-0.8, -1e-3),
              nre=200, nim=100, N=3000, chunk=800):
    """log10|W| on a grid using the vectorised continued fraction (method 1), and the
    list of local minima (QNM candidates)."""
    from .exterior import up_logderiv_cf_grid, rw_to_zerilli, integrate_real
    wr = np.linspace(*re, nre)
    wi = np.linspace(*im, nim)
    WW = (wr[None, :] + 1j * wi[:, None]).ravel()
    R2 = max(R, 4.5 * M)
    L = np.empty(WW.shape, complex)
    for s in range(0, WW.size, chunk):
        L[s:s + chunk] = up_logderiv_cf_grid(WW[s:s + chunk], l, R2, M, N)
    G = np.empty(WW.shape)
    for k, w in enumerate(WW):
        P, dP = 1.0 + 0j, L[k]
        if parity == 'even':
            P, dP = rw_to_zerilli(R2, P, dP, l, w, M)
        if R2 > R:
            sol = integrate_real([P, dP], R2, R, l, parity, w, M)
            P, dP = sol.y[0, -1], sol.y[1, -1]
        yu = np.array([P, dP])
        yJ = y_junction(w, l, parity, R, M, model, vs2)
        W = (yJ[0] * yu[1] - yJ[1] * yu[0]) / (np.linalg.norm(yJ) * np.linalg.norm(yu))
        G[k] = np.log10(abs(W) + 1e-300)
    G = G.reshape(nim, nre)
    cands = []
    for i in range(1, nim - 1):
        for j in range(1, nre - 1):
            c = G[i, j]
            nb = G[i - 1:i + 2, j - 1:j + 2]
            if np.isfinite(c) and c <= np.nanmin(nb) and c < -0.5:
                cands.append(wr[j] + 1j * wi[i])
    return wr, wi, G, cands


def refine(cands, l, parity, R, M=1.0, model='C', vs2=0.0, tol_res=1e-8):
    """Polish candidates with both methods; keep converged, distinct roots."""
    out = []
    for g in cands:
        w1, r1, ok = find_qnm(g, l, parity, R, M, model, vs2, method=1)
        if not (ok and r1 < tol_res and w1.imag < 0 and w1.real > 0):
            continue
        if any(abs(w1 - o['w1']) < 1e-7 for o in out):
            continue
        w2, r2, ok2 = find_qnm(w1, l, parity, R, M, model, vs2, method=2)
        out.append(dict(w1=w1, res1=r1, w2=w2, res2=r2, diff=abs(w1 - w2)))
    return sorted(out, key=lambda o: o['w1'].real)


def scan(l, parity, R, M=1.0, model='C', vs2=0.0, re=(0.02, 2.0), im=(-0.8, -1e-4), nre=160, nim=90):
    """log10|W| on a grid (method 1) -> list of local minima as QNM candidates."""
    wr = np.linspace(*re, nre)
    wi = np.linspace(*im, nim)
    G = np.empty((nim, nre))
    for i, b in enumerate(wi):
        for j, a in enumerate(wr):
            try:
                G[i, j] = np.log10(abs(wronskian(a + 1j * b, l, parity, R, M, model, vs2, 1)) + 1e-300)
            except Exception:
                G[i, j] = np.nan
    cands = []
    for i in range(1, nim - 1):
        for j in range(1, nre - 1):
            c = G[i, j]
            if np.isfinite(c) and c < np.nanmin([G[i - 1, j], G[i + 1, j], G[i, j - 1], G[i, j + 1],
                                                  G[i - 1, j - 1], G[i + 1, j + 1], G[i - 1, j + 1],
                                                  G[i + 1, j - 1]]) and c < -1.0:
                cands.append(wr[j] + 1j * wi[i])
    return wr, wi, G, cands
