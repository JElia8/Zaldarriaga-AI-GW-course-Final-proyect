"""
Exterior (Schwarzschild, r > R) solutions of the Regge-Wheeler and Zerilli
equations  d^2Psi/dr*^2 + [w^2 - V(r)] Psi = 0,   e^{-i w t} convention,
Psi -> B_in e^{-i w r*} + B_out e^{+i w r*}  as r* -> +inf.

Three independent tools:
 (a) direct ODE integration in r (scipy DOP853, rtol=1e-10, atol=1e-12), also
     along complex rays r = r0 + s e^{i phi} for complex w (QNMs);
 (b) large-r asymptotic series of the outgoing solution
       Psi_up = e^{i w r*} sum_k c_k r^{-k},
       2 i w (k+1) c_{k+1} = [k(k+1) - l(l+1)] c_k - 2M (k^2 - 4) c_{k-1}   (RW)
     derived in the derivation document (Sec. 5.2) from Regge & Wheeler (1957) Eq. (24)-(25);
 (c) Leaver-type continued fraction for the purely outgoing RW solution about a
     point r = R2 (Leins, Nollert & Soffel 1993; Pani et al. 2009 App. B,
     Eqs. B3-B19, adapted to e^{-i w t}; recurrence verified in
     symbolic/exterior_series.py), valid for R2 >= 4M.
Zerilli solutions are obtained from RW ones by the Chandrasekhar transformation
  Z = [mu(mu+2) + 72 M^2 f/(r(mu r + 6M))] Psi + 12 M dPsi/dr*
(Chandrasekhar 1983 Sec. 4.26; verified symbolically in symbolic/exterior_series.py).
"""
import numpy as np
from scipy.integrate import solve_ivp
from .potentials import V as Vpot

RTOL, ATOL = 1e-10, 1e-12        # PROMPT-mandated DOP853 tolerances


# ----------------------------------------------------------------- ODE in r
def ode_rhs(l, parity, w, M=1.0):
    """y = (Psi, dPsi/dr):  Psi'' = [(V - w^2) Psi / f - f' Psi'] / f."""
    def rhs(r, y):
        f = 1.0 - 2.0 * M / r
        fp = 2.0 * M / r**2
        return [y[1], ((Vpot(r, l, parity, M) - w**2) * y[0] / f - fp * y[1]) / f]
    return rhs


def integrate_real(y0, r0, r1, l, parity, w, M=1.0, dense=False, rtol=RTOL, atol=ATOL):
    """Integrate (Psi, dPsi/dr) along the real r axis from r0 to r1."""
    sol = solve_ivp(ode_rhs(l, parity, w, M), (r0, r1), np.asarray(y0, dtype=complex),
                    method='DOP853', rtol=rtol, atol=atol, dense_output=dense)
    return sol


def integrate_ray(y0, r_start, r_end, l, parity, w, M=1.0, rtol=RTOL, atol=ATOL):
    """Integrate along the straight segment from complex r_start to r_end.
    Parametrise r = r_start + s (r_end - r_start), s in [0,1]."""
    d = r_end - r_start
    rhs0 = ode_rhs(l, parity, w, M)

    def rhs(s, y):
        rr = r_start + s * d
        dy = rhs0(rr, y)
        return [d * dy[0], d * dy[1]]

    sol = solve_ivp(rhs, (0.0, 1.0), np.asarray(y0, dtype=complex), method='DOP853',
                    rtol=rtol, atol=atol)
    return sol.y[:, -1]


# ----------------------------------------------------------------- asymptotic series
def rstar_c(r, M=1.0):
    """Tortoise coordinate for complex r (principal branch of the log)."""
    return r + 2.0 * M * np.log(r / (2.0 * M) - 1.0 + 0j)


def up_asymptotic_RW(r, l, w, M=1.0, kmax=60):
    """Outgoing RW solution e^{i w r*} sum c_k r^{-k} and d/dr at (possibly complex) r.
    Series truncated at its smallest term (asymptotic, not convergent)."""
    lam = l * (l + 1)
    c = [1.0 + 0j]
    cm1 = 0.0
    for k in range(0, kmax):
        ckp1 = ((k * (k + 1) - lam) * c[k] - 2 * M * (k * k - 4) * (c[k - 1] if k >= 1 else cm1)) \
            / (2j * w * (k + 1))
        c.append(ckp1)
    terms = np.array([c[k] * r**(-k) for k in range(kmax + 1)])
    # truncate at the smallest (non-vanishing) term; for l=2, c_3 = 0 identically
    mags = np.abs(terms)
    mz = np.where(mags == 0, np.inf, mags)
    kstop = int(np.argmin(mz[1:]) + 1)
    u = np.sum(terms[:kstop])
    du = np.sum([-k * c[k] * r**(-k - 1) for k in range(kstop)])
    f = 1.0 - 2.0 * M / r
    ph = np.exp(1j * w * rstar_c(r, M))
    Psi = ph * u
    dPsi = ph * (1j * w / f * u + du)
    return Psi, dPsi, mags[kstop]


def rw_to_zerilli(r, Psi, dPsi, l, w, M=1.0):
    """Chandrasekhar map (Psi_RW, dPsi_RW/dr) -> (Z, dZ/dr) at r (complex ok)."""
    mu = (l - 1) * (l + 2)
    kap = mu * (mu + 2)
    f = 1.0 - 2.0 * M / r
    fp = 2.0 * M / r**2
    A = kap + 72 * M**2 * f / (r * (mu * r + 6 * M))
    # dA/dr
    g = r * (mu * r + 6 * M)
    dA = 72 * M**2 * (fp * g - f * (2 * mu * r + 6 * M)) / g**2
    V = Vpot(r, l, 'odd', M)
    d2Psi = ((V - w**2) * Psi / f - fp * dPsi) / f
    Z = A * Psi + 12 * M * f * dPsi
    dZ = dA * Psi + A * dPsi + 12 * M * (fp * dPsi + f * d2Psi)
    return Z, dZ


def up_asymptotic(r, l, parity, w, M=1.0):
    """Outgoing solution (RW or Zerilli) at large |r| via the asymptotic series,
    normalised in both cases as Psi_up -> e^{i w r*} (the Chandrasekhar map
    multiplies the amplitude by mu(mu+2) + 12 i M w, which is divided out)."""
    P, dP, err = up_asymptotic_RW(r, l, w, M)
    if parity == 'odd':
        return P, dP, err
    Z, dZ = rw_to_zerilli(r, P, dP, l, w, M)
    mu = (l - 1) * (l + 2)
    nrm = mu * (mu + 2) + 12j * M * w
    return Z / nrm, dZ / nrm, err


# ----------------------------------------------------------------- continued fraction
def up_logderiv_cf(w, l, R2, M=1.0, N=4000, tol=1e-14):
    """(dPsi/dr)/Psi of the purely outgoing RW solution at r = R2 (R2 >= 4M).

    Psi = chi(r) phi(z), z = 1 - R2/r, chi = (r-2M)^{2iMw} e^{iwr}; phi = sum a_n z^n.
    4-term recurrence  alpha_n a_{n+1} + beta_n a_n + gamma_n a_{n-1} + delta_n a_{n-2} = 0
      alpha_n = n(n+1) c0, beta_n = n(n-1) c1 + n d0,
      gamma_n = (n-1)(n-2) c2 + (n-1) d1 + e0, delta_n = (n-2)(n-3) c3 + (n-2) d2 + e1
    [Pani et al. 2009 Eq. (B7) with w -> -w], reduced to 3 terms by Gaussian
    elimination (their Eqs. B10-B13, Leaver 1990), minimal solution through the
    continued fraction a1/a0 = -g1/(b1 - a1 g2/(b2 - ...)) (their Eq. B17).
    Then  Psi'/Psi = chi'/chi + (a1/a0)/R2,  chi'/chi = i w R2/(R2 - 2M).
    """
    lam = l * (l + 1)
    m = M / R2
    c0, c1, c2, c3 = 1 - 2 * m, 6 * m - 2, 1 - 6 * m, 2 * m
    d0, d1, d2 = -2 + 6 * m + 2j * w * R2, 2 - 12 * m, 6 * m
    e0, e1 = 6 * m - lam, -6 * m

    def coeffs(n):
        al = n * (n + 1) * c0
        be = n * (n - 1) * c1 + n * d0
        ga = (n - 1) * (n - 2) * c2 + (n - 1) * d1 + e0
        de = (n - 2) * (n - 3) * c3 + (n - 2) * d2 + e1
        return al, be, ga, de

    def cf(N):
        # Gaussian elimination forward
        ah = np.zeros(N + 2, complex)
        bh = np.zeros(N + 2, complex)
        gh = np.zeros(N + 2, complex)
        for n in range(1, N + 2):
            al, be, ga, de = coeffs(n)
            if n == 1:
                ah[n], bh[n], gh[n] = al, be, ga
            else:
                ah[n] = al
                bh[n] = be - ah[n - 1] * de / gh[n - 1]
                gh[n] = ga - bh[n - 1] * de / gh[n - 1]
        # backward evaluation of the continued fraction for a_n/a_{n-1}
        # a_n/a_{n-1} = -gh_n / (bh_n + ah_n * a_{n+1}/a_n)
        ratio = 0.0 + 0j
        for n in range(N + 1, 0, -1):
            ratio = -gh[n] / (bh[n] + ah[n] * ratio)
        return ratio          # = a1/a0

    # adaptive depth: double until the continued fraction has converged
    n = 200
    r1 = cf(n)
    while True:
        n *= 2
        r2 = cf(n)
        if abs(r2 - r1) <= tol * 100 * abs(r2) or n >= 128000:
            break
        r1 = r2
    a1a0 = r2
    return 1j * w * R2 / (R2 - 2 * M) + a1a0 / R2


def up_logderiv_cf_grid(W, l, R2, M=1.0, N=3000):
    """Vectorised version of up_logderiv_cf over an array of complex w (fixed depth N)."""
    W = np.asarray(W, dtype=complex)
    lam = l * (l + 1)
    m = M / R2
    c0, c1, c2, c3 = 1 - 2 * m, 6 * m - 2, 1 - 6 * m, 2 * m
    d0 = -2 + 6 * m + 2j * W * R2
    d1, d2 = 2 - 12 * m, 6 * m
    e0, e1 = 6 * m - lam, -6 * m
    ah = np.zeros((N + 2,) + W.shape, complex)
    bh = np.zeros_like(ah)
    gh = np.zeros_like(ah)
    for n in range(1, N + 2):
        al = n * (n + 1) * c0
        be = n * (n - 1) * c1 + n * d0
        ga = (n - 1) * (n - 2) * c2 + (n - 1) * d1 + e0
        de = (n - 2) * (n - 3) * c3 + (n - 2) * d2 + e1
        if n == 1:
            ah[n], bh[n], gh[n] = al, be, ga
        else:
            ah[n] = al
            bh[n] = be - ah[n - 1] * de / gh[n - 1]
            gh[n] = ga - bh[n - 1] * de / gh[n - 1]
    ratio = np.zeros(W.shape, complex)
    for n in range(N + 1, 0, -1):
        ratio = -gh[n] / (bh[n] + ah[n] * ratio)
    return 1j * W * R2 / (R2 - 2 * M) + ratio / R2


def up_at_R_cf(w, l, parity, R, M=1.0, N=4000):
    """(Psi_up, dPsi_up/dr) at r = R (arbitrary normalisation) using the
    continued fraction at R2 = max(R, 4.5M) and, if R < R2, a short inward
    DOP853 integration from R2 to R (along the real axis)."""
    R2 = max(R, 4.5 * M)
    L = up_logderiv_cf(w, l, R2, M, N)
    P, dP = 1.0 + 0j, L
    if parity == 'even':
        P, dP = rw_to_zerilli(R2, P, dP, l, w, M)
    if R2 > R:
        sol = integrate_real([P, dP], R2, R, l, parity, w, M)
        P, dP = sol.y[0, -1], sol.y[1, -1]
    return np.array([P, dP])
