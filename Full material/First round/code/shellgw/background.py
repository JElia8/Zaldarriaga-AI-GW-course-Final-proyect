"""
Background shell models (units G = c = 1, M = 1 by default).

The shell sits at areal radius R with flat interior and Schwarzschild exterior.
Angular Israel condition (independent of the shell's equation of state):
    [K^theta_theta] = (sqrt(f) - 1)/R = -4 pi sigma          (Ipser & Sikivie 1984 Eqs. 3.5, 3.6,
                                                              2.6; Poisson 2004 Eq. 3.80)
so that at Rdot = 0:    4 pi sigma R = 1 - sqrt(f(R)),  f = 1 - 2M/R.

MODEL C (primary, truly static) -- static perfect-fluid shell:
    S^ab = sigma u^a u^b + p (h^ab + u^a u^b)       (Poisson 2004 Eq. 3.76; I&S Eq. 2.9 with p = -tau)
    sigma = (1 - sqrt f)/(4 pi R)                    (Poisson Eq. 3.80)
    p     = (1 - M/R - sqrt f)/(8 pi R sqrt f)       (Poisson Eq. 3.81)
    kappa = -tau/sigma = p/sigma = (1 - sqrt f)/(4 sqrt f)
  Equivalently I&S Eq. (3.7a) with Rdot = Rddot = 0 requires tau/sigma = -kappa < 0.
  kappa is NOT a free parameter: it is fixed by R/M.  Dominant energy condition
  p <= sigma  <=>  sqrt f >= 1/5  <=>  R >= 25M/12.

MODEL A (adiabatic) -- pure domain wall tau = sigma, S^ab = -sigma h^ab (I&S Eq. 2.10),
  frozen at its turning point R = R_m, Rdot = 0:
    M = 4 pi sigma R_m^2 (1 - 2 pi sigma R_m)       (I&S Eq. 3.9)  [identical to 4 pi sigma R = 1 - sqrt f]
    Rddot = -M/(R^2 (1 + sqrt f)) - 2 sqrt f / R      (I&S Eq. 3.7a with alpha = 1, beta = sqrt f,
                                                       tau/sigma = 1; proper-time derivative)
  Always Rddot < 0: collapse (I&S discussion after Eq. 3.7a).

NOTE on "Option B" (dust shell, tau = 0) of the task description: I&S Eq. (3.7a) with
tau = 0 gives (alpha+beta) Rddot = -alpha M / R^2 < 0, so a static dust shell does NOT
exist for M != 0 (Poisson 2004 Sec. 3.9, Eq. 3.70 gives the same: dust shells always
move).  The formula M = 4 pi sigma R^2 sqrt f quoted there does not follow from the
junction conditions; at a dust-shell turning point one has instead
M = 4 pi sigma R^2 (1 - 2 pi sigma R) (Poisson Eq. 3.70 with Rdot = 0), i.e. exactly
the same momentarily-static geometry as model A.
"""
import numpy as np


def sqrtf(R, M=1.0):
    return np.sqrt(1.0 - 2.0 * M / R)


def sigma(R, M=1.0):
    """Surface energy density, Poisson (2004) Eq. (3.80) = I&S Eq. (3.9) at Rdot=0."""
    return (1.0 - sqrtf(R, M)) / (4.0 * np.pi * R)


def pressure_static(R, M=1.0):
    """Surface pressure of the static fluid shell, Poisson (2004) Eq. (3.81)."""
    s = sqrtf(R, M)
    return (1.0 - M / R - s) / (8.0 * np.pi * R * s)


def kappa_static(R, M=1.0):
    """kappa = -tau/sigma = p/sigma = (1 - sqrt f)/(4 sqrt f) for static equilibrium."""
    s = sqrtf(R, M)
    return (1.0 - s) / (4.0 * s)


def mass_turning_point(sig, Rm):
    """I&S Eq. (3.9): M = 4 pi sigma Rm^2 (1 - 2 pi sigma Rm)."""
    return 4 * np.pi * sig * Rm**2 * (1 - 2 * np.pi * sig * Rm)


def Rddot_domain_wall(R, M=1.0):
    """Proper-time acceleration of a domain wall at its turning point, I&S Eq. (3.7a)."""
    s = sqrtf(R, M)
    return -M / (R**2 * (1 + s)) - 2 * s / R


def static_check_IS37a(R, tau_over_sigma, M=1.0):
    """Residual of I&S Eq. (3.7a) at Rdot = Rddot = 0:  -M/R^2 - 2(tau/sigma) beta (1+beta)/R."""
    b = sqrtf(R, M)
    return -M / R**2 - 2 * tau_over_sigma * b * (1 + b) / R


def dynamical_frequency(R, M=1.0):
    """Shell dynamical rate sqrt(|Rddot|/R) (proper time) converted to exterior-time
    frequency (multiply by sqrt f): the adiabatic model A requires omega >> this."""
    return np.sqrt(abs(Rddot_domain_wall(R, M)) / R) * sqrtf(R, M)
