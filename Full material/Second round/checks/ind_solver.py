"""
Independent (second-round) thin-shell solver.

Differences from the first-round code, on purpose:
  * interior Riccati-Bessel/Hankel functions from mpmath (not scipy.special);
  * exterior ODE integrated with LSODA on the real 4-vector (not DOP853);
  * Zerilli asymptotic series obtained DIRECTLY from the 1/r expansion of V_Z/f (general
    recursion), with no Chandrasekhar map; RW uses the same general recursion;
  * QNMs: outgoing solution from the asymptotic series on the complex ray
    r = R + s e^{i phi} with phi = pi/2 - arg(w) - 0.35 (different ray), integrated inward;
    roots with mpmath.findroot (secant/Muller) instead of scipy.optimize.root;
  * the odd junction is the (independently confirmed) matrix T_odd; the even junction
    is taken from the first round's generated module (it is the object under test).
"""
import sys
import os
import numpy as np
import mpmath as mp
import sympy as sp
from scipy.integrate import solve_ivp

M = 1.0
_FR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fr_code_pristine')


# ----------------------------------------------------------------- potentials
def Vpot(r, l, parity):
    f = 1 - 2*M/r
    if parity == 'odd':
        return f*(l*(l + 1)/r**2 - 6*M/r**3)
    n = (l - 1)*(l + 2)/2
    return f*(2*n*n*(n + 1)*r**3 + 6*n*n*M*r*r + 18*n*M*M*r + 18*M**3)/(r**3*(n*r + 3*M)**2)


_vcache = {}


def v_coeffs(l, parity, J=80):
    """coefficients v_j of V/f = sum_j v_j r^-j (M = 1), exact via sympy."""
    key = (l, parity, J)
    if key in _vcache:
        return _vcache[key]
    u = sp.Symbol('u')           # u = 1/r
    if parity == 'odd':
        e = l*(l + 1)*u**2 - 6*u**3
    else:
        n = sp.Rational((l - 1)*(l + 2), 2)
        e = (2*n**2*(n + 1)*u**-3 + 6*n**2*u**-2 + 18*n*u**-1 + 18)/(u**-3*(n/u + 3)**2)
        e = sp.simplify(e)
    ser = sp.series(e, u, 0, J + 1).removeO()
    v = [float(ser.coeff(u, j)) for j in range(J + 1)]
    _vcache[key] = v
    return v


def up_series(r, l, parity, w, kmax=60):
    """Outgoing solution e^{i w r*} sum c_k r^-k (and d/dr) at complex/real r."""
    v = v_coeffs(l, parity)
    c = [1.0 + 0j]
    for k in range(0, kmax):
        s = k*(k + 1)*c[k]
        if k >= 1:
            s -= 2*M*(k - 1)*(k + 1)*c[k - 1]
        for j in range(2, min(len(v), k + 3)):
            s -= v[j]*c[k + 2 - j]
        c.append(s/(2j*w*(k + 1)))
    terms = np.array([c[k]*r**(-k) for k in range(kmax + 1)])
    mags = np.abs(terms)
    stop = kmax
    for k in range(4, kmax):
        if mags[k] > mags[k - 1] and mags[k - 1] > 0:
            stop = k - 1
            break
    u = terms[:stop].sum()
    du = sum(-k*c[k]*r**(-k - 1) for k in range(stop))
    rs = r + 2*M*np.log(r/(2*M) - 1 + 0j)
    ph = np.exp(1j*w*rs)
    f = 1 - 2*M/r
    return ph*u, ph*(1j*w/f*u + du)


# ----------------------------------------------------------------- ODE (complex as real)
def integrate(y0, r0, r1, l, parity, w, rtol=1e-11, atol=1e-30):
    """Integrate Psi'' = [(V - w^2) Psi/f - f' Psi']/f along the straight segment r0 -> r1
    (both possibly complex) with LSODA on the real 4-vector."""
    d = r1 - r0

    def rhs(s, Y):
        rr = r0 + s*d
        P = Y[0] + 1j*Y[1]
        dP = Y[2] + 1j*Y[3]
        f = 1 - 2*M/rr
        d2 = ((Vpot(rr, l, parity) - w*w)*P/f - 2*M/rr**2*dP)/f
        a = d*dP
        b = d*d2
        return [a.real, a.imag, b.real, b.imag]
    Y0 = [y0[0].real, y0[0].imag, y0[1].real, y0[1].imag]
    sol = solve_ivp(rhs, (0.0, 1.0), Y0, method='LSODA', rtol=rtol, atol=atol)
    Y = sol.y[:, -1]
    return np.array([Y[0] + 1j*Y[1], Y[2] + 1j*Y[3]])


# ----------------------------------------------------------------- interior (mpmath)
def _sph(l, z, kind):
    fn = mp.besselj if kind == 'j' else mp.bessely
    return mp.sqrt(mp.pi/(2*z))*fn(l + mp.mpf(1)/2, z)


def rb_j(l, z):
    """Riccati-Bessel z j_l(z) and derivative, (z j_l)' = z j_{l-1} - l j_l (DLMF 10.51.2)."""
    z = mp.mpc(z)
    jl, jm = _sph(l, z, 'j'), _sph(l - 1, z, 'j')
    return complex(z*jl), complex(z*jm - l*jl)


def rb_y(l, z):
    z = mp.mpc(z)
    yl, ym = _sph(l, z, 'y'), _sph(l - 1, z, 'y')
    return complex(z*yl), complex(z*ym - l*yl)


def interior(l, w, R, kind):
    s = np.sqrt(1 - 2*M/R)
    nu = w/s
    j, jp = rb_j(l, nu*R)
    if kind == 'regular':
        return np.array([j, nu*jp])
    y, yp = rb_y(l, nu*R)
    return np.array([-y - 1j*j, nu*(-yp - 1j*jp)])       # ingoing Riccati-Hankel


# ----------------------------------------------------------------- junction
def T_odd(R):
    s = np.sqrt(1 - 2*M/R)
    return np.array([[1, 0], [(1 - s)/(R*s), 1/s]], dtype=complex)


def T_even(w, l, R, model, vs2):
    if _FR not in sys.path:
        sys.path.insert(0, _FR)
    from shellgw.junction import T_even as Tf           # first-round generated matrix (under test)
    return Tf(w, l, R, 1.0, model, vs2)


def vs2_of(R, Gamma=2.0):
    s = np.sqrt(1 - 2*M/R)
    k = (1 - s)/(4*s)
    return Gamma*k/(1 + k)


def transfer(parity, w, l, R, model='C', Gamma=2.0):
    if parity == 'odd':
        return T_odd(R)
    return T_even(w, l, R, model, vs2_of(R, Gamma) if model == 'C' else 0.0)


# ----------------------------------------------------------------- scattering
def RT(w, l, parity, R, model='C', Gamma=2.0, rfar=None):
    y = transfer(parity, w, l, R, model, Gamma) @ interior(l, w, R, 'ingoing')
    if rfar is None:
        rfar = max(400.0, 120.0/w)
    y = integrate(y, R, rfar, l, parity, w, atol=1e-14*np.abs(y).max())
    Pu, dPu = up_series(rfar, l, parity, w)
    A = np.array([[np.conj(Pu), Pu], [np.conj(dPu), dPu]])
    Bin, Bout = np.linalg.solve(A, y)
    return abs(Bout/Bin)**2, 1/abs(Bin)**2


# ----------------------------------------------------------------- QNMs
def up_at_R(w, l, parity, R, L=None, dphi=0.35):
    phi = np.pi/2 - np.angle(w) - dphi
    phi = min(max(phi, 0.15), 1.9)
    if L is None:
        L = max(70.0, 50.0/abs(w))
    rfar = R + L*np.exp(1j*phi)
    P, dP = up_series(rfar, l, parity, w)
    nrm = abs(P) + abs(dP)
    return integrate(np.array([P/nrm, dP/nrm]), rfar, R + 0j, l, parity, w)


def wronsk(w, l, parity, R, model='C', Gamma=2.0, **kw):
    w = complex(w)
    yJ = transfer(parity, w, l, R, model, Gamma) @ interior(l, w, R, 'regular')
    yu = up_at_R(w, l, parity, R, **kw)
    return (yJ[0]*yu[1] - yJ[1]*yu[0])/(np.linalg.norm(yJ)*np.linalg.norm(yu))


def qnm(guess, l, parity, R, model='C', Gamma=2.0, tol=1e-12, maxit=60, **kw):
    """secant iteration on the normalised Wronskian, stopped on |step| < tol |w|"""
    F = lambda z: wronsk(z, l, parity, R, model, Gamma, **kw)
    z0 = complex(guess)
    z1 = z0*(1 + 1e-6) + 1e-7j
    f0, f1 = F(z0), F(z1)
    for _ in range(maxit):
        if f1 == f0:
            break
        z2 = z1 - f1*(z1 - z0)/(f1 - f0)
        z0, f0 = z1, f1
        z1, f1 = z2, F(z2)
        if abs(z1 - z0) < tol*max(abs(z1), 1e-3):
            return z1
    raise RuntimeError('secant did not converge from %s' % guess)
