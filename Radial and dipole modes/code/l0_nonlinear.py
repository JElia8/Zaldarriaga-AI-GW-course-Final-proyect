"""
Nonlinear radial (l = 0) evolution of a perturbed static shell: does it oscillate or collapse?

Exact shell dynamics (Ipser & Sikivie 1984, Eqs. 3.7-3.9):
    Rdot^2 + V(R) = 0,   V(R) = 1 - [M/m(R) + m(R)/(2R)]^2,   m(R) = 4 pi R^2 sigma(R)
The equation of state is a polytrope in the rest-mass surface density mu (PSP26 Eq. 3.22):
    p = K mu^Gamma,   sigma = mu + p/(Gamma - 1)        (first law: dsigma = (sigma + p) dmu/mu)
and rest mass is conserved: mu(R) = mu_0 R_0^2/R^2.
K and mu_0 are chosen so that the static shell at R_0 (sigma_0, p_0 = kappa sigma_0) is an equilibrium.
The shell is then released from rest at R_0 (1 + delta): this changes M slightly (same matter, different energy).
We integrate  Rddot = -V'(R)/2  (proper time) and the exterior time dt/dtau = sqrt(f + Rdot^2)/f.
Units M_eq = 1 (mass of the unperturbed equilibrium).
Output: ../results/l0_nonlinear.json (curves used by make_figures.py).
"""
import json
import numpy as np
from scipy.integrate import solve_ivp


def shell(R0, Gam):
    x0 = np.sqrt(1 - 2/R0)
    sig0 = (1 - x0)/(4*np.pi*R0)
    p0 = (1 - x0)/(4*x0)*sig0
    mu0 = sig0 - p0/(Gam - 1)
    K = p0/mu0**Gam
    def m_of(R):
        mu = mu0*R0**2/R**2
        return 4*np.pi*R**2*(mu + K*mu**Gam/(Gam - 1))
    return m_of, mu0*4*np.pi*R0**2


def V(R, M, m_of):
    m = m_of(R)
    return 1 - (M/m + m/(2*R))**2


def dV(R, M, m_of, h=1e-6):
    return (V(R*(1 + h), M, m_of) - V(R*(1 - h), M, m_of))/(2*R*h)


def evolve(R0, Gam, delta, tau_max, Rstop=2.0):
    m_of, mrest = shell(R0, Gam)
    Rs = R0*(1 + delta)
    m = m_of(Rs)
    M = m*(1 - m/(2*Rs))                      # released from rest at Rs:  M/m + m/(2R) = 1
    def rhs(tau, y):
        R, Rd, t = y
        f = 1 - 2/R*1.0*M
        return [Rd, -dV(R, M, m_of)/2, np.sqrt(max(f + Rd**2, 0))/f if f > 0 else np.inf]
    def horizon(tau, y):
        return y[0] - 2*M*(1 + 1e-4)
    horizon.terminal = True
    sol = solve_ivp(rhs, (0, tau_max), [Rs, 0.0, 0.0], method='DOP853', rtol=1e-11, atol=1e-13,
                    events=horizon, dense_output=True, max_step=tau_max/4000)
    return sol, M, m_of, mrest


out = {}
if __name__ == '__main__':
    Gam = 2.0
    # linear prediction (l0_potential.py): omega0^2 R^3/M = 2x^2/(1+x) (Gamma - Gamma_1),  Omega = omega/x
    def lin_rate(R0):
        x = np.sqrt(1 - 2/R0)
        G1 = (1 + 2*x + 3*x**2)/(4*x**2)
        w2 = 2*x**2/(1 + x)*(Gam - G1)/R0**3
        return w2, w2/x**2                     # (exterior-time omega^2, proper-time Omega^2) with M = 1
    cases = {'R3_in': (3.0, -1e-3, 400.0), 'R3_out': (3.0, +1e-3, 3000.0),
             'R6_in': (6.0, -1e-2, 1500.0), 'R6_out': (6.0, +1e-2, 1500.0)}
    for name, (R0, dl, tmax) in cases.items():
        sol, M, m_of, mrest = evolve(R0, Gam, dl, tmax)
        tt = np.linspace(0, sol.t[-1], 3000)
        Y = sol.sol(tt)
        rec = {'R0': R0, 'delta': dl, 'M': M, 'rest_mass': mrest, 'binding': mrest - M,
               'tau': tt.tolist(), 'R': Y[0].tolist(), 't': Y[2].tolist(),
               'hit_horizon': bool(sol.status == 1), 'tau_end': float(sol.t[-1])}
        w2, O2 = lin_rate(R0)
        rec['lin_Omega2_proper'] = O2
        if O2 > 0:
            # measure the oscillation period from successive minima of R - R0
            Rr = Y[0] - R0
            idx = np.where((np.diff(np.sign(np.diff(Rr))) > 0))[0] + 1
            if len(idx) > 2:
                T = np.mean(np.diff(tt[idx]))
                rec['measured_Omega_proper'] = 2*np.pi/T
                rec['linear_Omega_proper'] = np.sqrt(O2)
        else:
            rec['linear_growth_proper'] = np.sqrt(-O2)
        # potential curve for the figure
        Rg = np.linspace(2.0*M*1.0001, 12.0 if R0 < 5 else 12.0, 1500)
        rec['Vgrid_R'] = Rg.tolist()
        rec['Vgrid_V'] = [V(r, M, m_of) for r in Rg]
        out[name] = rec
        print('%-7s R0=%.1f delta=%+.0e  M=%.8f  rest mass=%.6f  horizon=%s  tau_end=%.1f  t_end=%.1f'
              % (name, R0, dl, M, mrest, rec['hit_horizon'], rec['tau_end'], Y[2][-1]),
              {k: round(v, 6) for k, v in rec.items() if k.startswith(('measured', 'linear'))})
        # outer turning point, if any
        print('         R range reached: [%.4f, %.4f]' % (min(Y[0]), max(Y[0])))
    # equilibria on the polytropic sequence through the R0 = 3M shell (same K, same rest mass): V = V' = 0
    json.dump(out, open('../results/l0_nonlinear.json', 'w'))
