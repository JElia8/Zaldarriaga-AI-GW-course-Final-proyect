"""
CHECK 03b -- numerical thick-shell -> thin-shell limit of the odd-parity junction.

A smooth anisotropic shell with p_r = 0 (so that, by the anisotropic TOV equation,
p_t = rho r Phi'/2) occupies R - eps < r < R + eps, with mass M = 1.  In the thin limit
its surface density and tangential pressure tend EXACTLY to model C's sigma and p
(shown analytically in the verification report).  The axial master variable
X = e^{Phi-Lambda} h1/r obeys  X_{r*r*} + [w^2 - V] X = 0 with
V = e^{2Phi}[l(l+1)/r^2 - 6m/r^3 + 4 pi rho]  (derived in c03_axial_thick_shell_derive.py).

We integrate X from the flat interior (X = jhat_l(nu r), nu = w e^{-Phi_in}) through the
shell and compare the log-derivative direction of (X, dX/dr) at r = R + eps [sin of the angle] with the thin-shell
prediction built from the first round's junction  T_odd = [[1,0],[(1-s)/(R s), 1/s]]
(and, for contrast, three WRONG alternatives).  The mismatch must -> 0 as eps -> 0.
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.special import spherical_jn

M, R, l = 1.0, 3.0, 2
lam = l*(l + 1)


def make_shell(eps):
    shape = lambda r: np.where(np.abs(r - R) < eps, (1 - ((r - R)/eps)**2)**2, 0.0)
    norm = quad(lambda r: 4*np.pi*r*r*shape(r), R - eps, R + eps, epsabs=1e-14, epsrel=1e-13)[0]
    rho = lambda r: M*shape(r)/norm
    rr = np.linspace(R - eps, R + eps, 4001)
    # mass function by integration (cumulative Simpson via solve_ivp)
    sm = solve_ivp(lambda r, y: [4*np.pi*r*r*rho(r)], (R - eps, R + eps), [0.0], dense_output=True,
                   rtol=1e-13, atol=1e-15, max_step=eps/200)
    m = lambda r: 0.0 if r <= R - eps else (M if r >= R + eps else sm.sol(r)[0])
    # Phi: integrate inward from R+eps where Phi = 1/2 ln(1 - 2M/r)
    sP = solve_ivp(lambda r, y: [m(r)/(r*(r - 2*m(r)))], (R + eps, R - eps), [0.5*np.log(1 - 2*M/(R + eps))],
                   dense_output=True, rtol=1e-13, atol=1e-15, max_step=eps/200)
    Phi = lambda r: sP.sol(r)[0]
    return rho, m, Phi


def jhat(x):
    return x*spherical_jn(l, x), spherical_jn(l, x) + x*spherical_jn(l, x, derivative=True)


def thick_logderiv(w, eps):
    rho, m, Phi = make_shell(eps)
    Phin = Phi(R - eps)
    nu = w*np.exp(-Phin)
    j, jp = jhat(nu*(R - eps))
    y0 = [j, nu*jp]

    def rhs(r, y):
        mm, P = m(r), Phi(r)
        mp_ = 4*np.pi*r*r*rho(r)
        Pp = mm/(r*(r - 2*mm))
        Lp = (mp_*r - mm)/(r*(r - 2*mm))
        e2 = np.exp(-2*P)/(1 - 2*mm/r)                        # e^{2(Lambda - Phi)}
        V = np.exp(2*P)*(lam/r**2 - 6*mm/r**3 + 4*np.pi*rho(r))
        return [y[1], -(Pp - Lp)*y[1] - e2*(w*w - V)*y[0]]
    sol = solve_ivp(rhs, (R - eps, R + eps), y0, method='DOP853', rtol=1e-12, atol=1e-14, max_step=eps/100)
    return sol.y[:, -1], Phin


def thin_logderiv(w, eps, Tfun):
    s = np.sqrt(1 - 2*M/R)
    nu = w/s
    j, jp = jhat(nu*(R - eps))
    # flat interior up to R
    def rin(r, y):
        return [y[1], (lam/r**2 - nu*nu)*y[0]]
    y = solve_ivp(rin, (R - eps, R), [j, nu*jp], method='DOP853', rtol=1e-12, atol=1e-14).y[:, -1]
    y = Tfun(s) @ y
    def rout(r, y):
        f = 1 - 2*M/r
        V = f*(lam/r**2 - 6*M/r**3)
        return [y[1], ((V - w*w)*y[0]/f - 2*M/r**2*y[1])/f]
    y = solve_ivp(rout, (R, R + eps), y, method='DOP853', rtol=1e-12, atol=1e-14).y[:, -1]
    return y


cands = {
    'first round  (Delta = 4 pi sigma sqrt f)': lambda s: np.array([[1, 0], [(1 - s)/(R*s), 1/s]]),
    'wrong: no delta (Delta = 0)':               lambda s: np.array([[1, 0], [0, 1/s]]),
    'wrong: Delta = 4 pi sigma (no sqrt f)':     lambda s: np.array([[1, 0], [(1 - s)/(R*s*s), 1/s]]),
    'wrong: Delta = 2x first round':             lambda s: np.array([[1, 0], [2*(1 - s)/(R*s), 1/s]]),
}
out = []
print('R = %gM, l = %d.  sin(angle) between thick and thin (X, dX/dr) at r = R + eps' % (R, l))
for w in (0.2, 0.5, 1.0, 2.0):
    print(' w M = %.2f' % w)
    for eps in (0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625):
        Lth, Phin = thick_logderiv(w, eps)
        row = dict(w=w, eps=eps, Phi_in_minus_half_ln_f=Phin - 0.5*np.log(1 - 2/R))
        txt = '   eps=%.4f  (Phi_in - ln sqrt f_R = %+.1e): ' % (eps, row['Phi_in_minus_half_ln_f'])
        for k, T in cands.items():
            v2 = thin_logderiv(w, eps, T)
            d = abs(Lth[0]*v2[1] - Lth[1]*v2[0])/(np.linalg.norm(Lth)*np.linalg.norm(v2))
            row[k] = d
            txt += ' %.2e' % d
        out.append(row)
        print(txt)
print(' columns:', ' | '.join(cands))
import json
with open('out_c03b.json', 'w') as fh:
    json.dump(out, fh, indent=1)
