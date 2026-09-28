"""
CHECK 01 -- background shell: independent symbolic re-derivation (second round).

Written from scratch (does not import any first-round code).  Uses only the
Israel conditions  [K^a_b] - delta^a_b [K] = -8 pi S^a_b  with the extrinsic
curvatures of a shell r = R(tau) between Minkowski (inside) and Schwarzschild
(outside) as given by Ipser & Sikivie (1984) Eqs. (3.5)-(3.6) / Poisson (2004)
Sec. 3.9:
   K^th_th(+) = beta/R,  K^th_th(-) = alpha/R,
   K^tau_tau(+) = (Rdd + M/R^2)/beta,   K^tau_tau(-) = Rdd/alpha.
Surface stress S^a_b = diag(-sigma, p, p).
"""
import sympy as sp

R, M, sig, p, Rd, Rdd, tau_ = sp.symbols('R M sigma p Rdot Rddot tau', real=True)
s = sp.symbols('s', positive=True)            # s = sqrt(f(R))

alpha = sp.sqrt(1 + Rd**2)
beta = sp.sqrt(1 - 2*M/R + Rd**2)
Ktt_p, Ktt_m = (Rdd + M/R**2)/beta, Rdd/alpha
Kth_p, Kth_m = beta/R, alpha/R
jtt, jth = Ktt_p - Ktt_m, Kth_p - Kth_m
jK = jtt + 2*jth
# Israel, mixed components
eq_tt = sp.Eq(jtt - jK, -8*sp.pi*(-sig))
eq_th = sp.Eq(jth - jK, -8*sp.pi*p)
sol = sp.solve([eq_tt, eq_th], [sig, p], dict=True)[0]
print('sigma =', sp.simplify(sol[sig]))
print('p     =', sp.simplify(sol[p]))

# ---------------- compare with Ipser & Sikivie Eq. (3.7a,b), with tau = -p ---------------
tauIS = -p
IS37a = (alpha+beta)*Rdd + alpha*M/R**2 + 2*tauIS/sig*alpha*beta*(alpha+beta)/R
IS37b = (alpha-beta)*Rdd + alpha*M/R**2 - 4*sp.pi*(sig - 2*tauIS)*alpha*beta
for nm, e in (('I&S (3.7a)', IS37a), ('I&S (3.7b)', IS37b)):
    r_ = sp.simplify(e.subs(sol))
    # numerical test at random point
    val = r_.subs({R: 3.7, M: 1.0, Rd: 0.31, Rdd: -0.2})
    print(nm, 'residual with Israel sigma,p (numeric):', sp.N(val))

# ---------------- mass formula I&S (3.8) and (3.9) ---------------
Mform = sp.Rational(1, 2)*(alpha+beta)*4*sp.pi*sol[sig]*R**2
print('I&S (3.8) M - formula at random point:', sp.N((Mform - M).subs({R: 3.7, M: 1.0, Rd: 0.31})))
# at Rdot=0: 4 pi sigma R = 1 - s
sig0 = sp.simplify(sol[sig].subs(Rd, 0))
print('Rdot=0: 4 pi sigma R =', sp.simplify(4*sp.pi*sig0*R))
Ms = R*(1 - s**2)/2                                    # M in terms of s
chk39 = sp.simplify((4*sp.pi*sig0*R**2*(1 - 2*sp.pi*sig0*R)).subs(M, Ms) - Ms)
print('I&S (3.9) M = 4 pi sigma R^2 (1 - 2 pi sigma R) residual:', chk39)

# ---------------- static shell (Option C) ---------------
p_stat = sp.simplify(sol[p].subs({Rd: 0, Rdd: 0}).subs(M, Ms))
sig_stat = sp.simplify(sig0.subs(M, Ms))
kappa = sp.simplify(p_stat/sig_stat)
print('static: sigma =', sig_stat, ' p =', p_stat, ' kappa = p/sigma =', sp.factor(kappa))
P04_381 = (1 - Ms/R - s)/(8*sp.pi*R*s)                  # Poisson (2004) Eq. (3.81)
print('Poisson (3.81) residual:', sp.simplify(p_stat - P04_381))
# kappa claimed = (1-s)/(4s) and = M/(2 R s (1+s))
print('kappa - (1-s)/(4s) =', sp.simplify(kappa - (1-s)/(4*s)),
      ';  kappa - M/(2Rs(1+s)) =', sp.simplify(kappa - Ms/(2*R*s*(1+s))))
# DEC p <= sigma  <=> kappa <= 1 <=> s >= 1/5 <=> R >= 25M/12
print('kappa=1 at s =', sp.solve(sp.Eq(kappa, 1), s), '-> R/M =', sp.nsimplify(1/(1 - sp.Rational(1, 25))*2))
# causality v_s^2 = Gamma kappa/(1+kappa) <= 1 for Gamma = 2
vs2 = 2*kappa/(1+kappa)
print('Gamma=2: v_s^2 = 1 at s =', sp.solve(sp.Eq(vs2, 1), s))

# ---------------- dust (Option B) ---------------
# static dust: p = 0 and Rdot = Rddot = 0
p0 = sp.simplify(sol[p].subs({Rd: 0, Rdd: 0}).subs(M, Ms))
print('dust: p(static) = 0  =>  s in', sp.solve(sp.Eq(p0, 0), s), '(only s = 1, i.e. M = 0)')
# the prompt's formula M = 4 pi sigma R^2 sqrt(f): which equation does it come from?
# pressure equation with p = 0 and Rdot = Rddot = 0 alone gives [K^tau_tau] = -[K^th_th] = 4 pi sigma:
e_p0 = sp.simplify((jtt + jth).subs({Rd: 0, Rdd: 0}))   # 8 pi p = [K^tt] + [K^thth]
print('p=0, static:  [K^tau_tau] =', sp.simplify(jtt.subs({Rd: 0, Rdd: 0})), ' = 4 pi sigma  ->  M = 4 pi sigma R^2 sqrt(f)')
print('   i.e. the task formula is the p=0 (tau-tau/pressure) equation; combined with the angular one',
      '4 pi sigma R = 1 - sqrt f it forces s = 1.')
# dust turning point acceleration
Rdd_dust = sp.solve(sp.Eq(sol[p].subs(Rd, 0), 0), Rdd)[0]
print('dust at turning point: Rddot =', sp.simplify(Rdd_dust.subs(M, Ms)),
      ' vs claimed -M/(R^2(1+s)):', sp.simplify((Rdd_dust + M/(R**2*(1+s))).subs(M, Ms)))

# ---------------- domain wall (Option A) at turning point ---------------
Rdd_wall = sp.solve(sp.Eq(sol[p].subs(Rd, 0), -sol[sig].subs(Rd, 0)), Rdd)[0]
Rdd_wall = sp.simplify(Rdd_wall.subs(M, Ms))
print('wall: Rddot =', Rdd_wall, ';  - (1+3s)/(2R) difference:', sp.simplify(Rdd_wall + (1 + 3*s)/(2*R)))
print('      IS 3.7a form -M/(R^2(1+s)) - 2s/R difference:',
      sp.simplify(Rdd_wall - (-Ms/(R**2*(1+s)) - 2*s/R)))
# numbers of Table (background.tex)
import math
print('\nR/M  sigma  p  kappa  vs2  Rdd_wall  w_dyn  sqrt(M/R^3)')
for Rv in (2.2, 3.0, 6.0, 10.0):
    sv = math.sqrt(1 - 2/Rv)
    sg = (1 - sv)/(4*math.pi*Rv)
    kp = (1 - sv)/(4*sv)
    rdd = -(1 + 3*sv)/(2*Rv)
    print('%5.1f %.5f %.3e %.4f %.4f %.4f %.4f %.4f' % (Rv, sg, kp*sg, kp, 2*kp/(1+kp), rdd,
                                                     sv*math.sqrt(abs(rdd)/Rv), Rv**-1.5))
# wall collapse from rest: proper time to reach r = 2M, for the table of timescales
