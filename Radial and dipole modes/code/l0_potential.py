"""
l = 0 (radial) oscillations of the static perfect-fluid shell, from the EXACT equation of motion.

Setting (same as the static fluid shell, "model C"): flat interior, Schwarzschild exterior of mass M,
thin shell at areal radius R with surface energy density sigma and surface pressure p.

Exact spherical dynamics (Ipser & Sikivie 1984, Eqs. 3.7-3.9; Poisson, Toolkit Sec. 3.9):
    sigma = [sqrt(1 + Rdot^2) - sqrt(f + Rdot^2)] / (4 pi R),       Rdot = dR/dtau (shell proper time)
    d(sigma A)/dtau + p dA/dtau = 0,   A = 4 pi R^2                   (first law, S^ab_{|b} = 0)
With m(R) = 4 pi R^2 sigma(R), squaring twice gives the energy equation
    Rdot^2 + V(R) = 0,     V(R) = 1 - [M/m + m/(2R)]^2 .
A static shell sits at V = V' = 0. A small radial displacement obeys
    d^2 dR/dtau^2 = -(V''/2) dR    ->   Omega_0^2 = V''(R0)/2   (proper time)
    omega_0^2 = f(R0) Omega_0^2                                   (exterior / Schwarzschild time)

Equation of state (PSP26 convention, the one used for the static shell throughout):
    delta p = Gamma p/(sigma+p) delta sigma,  i.e.  v_s^2 := dp/dsigma = Gamma p/(sigma+p).
The first law gives dsigma/dR = -2(sigma+p)/R along the motion, hence
    m'  = -8 pi R p,       m'' = -8 pi p + 16 pi v_s^2 (sigma+p).

Output: closed form of omega_0^2, the marginal index Gamma_crit(C) (compared with PSP26 Eq. 3.16),
and numbers written to ../results/l0_potential.json.
"""
import json
import sympy as sp

R, M, G, eps = sp.symbols('R M Gamma epsilon', positive=True)
x = sp.symbols('x', positive=True)          # x = sqrt(f(R)), 0 < x < 1
C = sp.symbols('C', positive=True)          # compactness M/R

f = 1 - 2*M/R
sq = sp.sqrt(f)

# ---- background (static shell): sigma, p = kappa sigma  (Poisson Toolkit Eqs. 3.80-3.81)
sig0 = (1 - sq)/(4*sp.pi*R)
kap = (1 - sq)/(4*sq)
p0 = kap*sig0
vs2 = G*p0/(sig0 + p0)

# ---- V(r) near R0 through second order: build m(r) as a Taylor series with the exact m', m''
m0 = 4*sp.pi*R**2*sig0
m1 = -8*sp.pi*R*p0
m2 = -8*sp.pi*p0 + 16*sp.pi*vs2*(sig0 + p0)
d = sp.symbols('d')                          # displacement r - R0
m_of = m0 + m1*d + m2*d**2/2
A_of = M/m_of + m_of/(2*(R + d))
V_of = 1 - A_of**2

V0 = sp.simplify(V_of.subs(d, 0))
V1 = sp.simplify(sp.diff(V_of, d).subs(d, 0))
V2 = sp.diff(V_of, d, 2).subs(d, 0)

# express in x = sqrt(f):  M = R(1 - x^2)/2
to_x = {M: R*(1 - x**2)/2}
V0x = sp.simplify(V0.subs(to_x))
V1x = sp.simplify(V1.subs(to_x))
V2x = sp.factor(sp.simplify(V2.subs(to_x)))
print('V(R0)   =', V0x, '   (must be 0: equilibrium)')
print("V'(R0)  =", V1x, '   (must be 0: equilibrium)')
print("V''(R0) =", V2x)

Om2_tau = sp.factor(sp.simplify(V2x/2))                   # proper-time frequency^2
om2 = sp.factor(sp.simplify(x**2*Om2_tau))                # exterior-time frequency^2
# natural units: omega^2 R^3 / M  (the matter-mode scale used for l >= 2)
vsig2 = sp.factor(sp.simplify(om2*R**3/(R*(1 - x**2)/2)))
print('\nomega_0^2 R^3/M  =', vsig2)

# ---- marginal stability
Gc = sp.solve(sp.numer(sp.together(vsig2)), G)
print('Gamma_crit(x) =', [sp.factor(g) for g in Gc])
# PSP26 Eq. (3.16b): Gamma_1 = (1 + 2 sqrt F + 3F)/(4F)
G1_psp = (1 + 2*x + 3*x**2)/(4*x**2)
print('Gamma_crit - Gamma_1(PSP26 3.16b) =', [sp.simplify(g - G1_psp) for g in Gc])

# write omega_0^2 R^3/M = K(x) (Gamma - Gamma_1(x))
Kx = sp.factor(sp.simplify(vsig2/(G - G1_psp)))
print('omega_0^2 R^3/M = K(x) (Gamma - Gamma_1),  K(x) =', Kx)
print('   check dGamma-independence of K:', sp.simplify(sp.diff(Kx, G)))

# Newtonian limit, x -> 1 (C -> 0): expect Gamma - 3/2  (PSP26 Sec. 10 H; LeMaitre & Poisson 2019)
xC = sp.sqrt(1 - 2*C)
ser = sp.series(vsig2.subs(x, xC), C, 0, 2).removeO()
print('\nsmall-C expansion of omega_0^2 R^3/M =', sp.expand(ser))

# ---- numbers
out = {'formula_omega0sq_R3_over_M': str(vsig2), 'K_x': str(Kx), 'Gamma1_x': str(G1_psp)}
fn = sp.lambdify((x, G), vsig2, 'mpmath')
g1 = sp.lambdify(x, G1_psp)
tab = []
for RM in (2.2, 2.5, 3.0, 4.0, 6.0, 10.0, 20.0, 100.0):
    xv = (1 - 2/RM)**0.5
    row = {'R_over_M': RM, 'C': 1/RM, 'Gamma1': g1(xv)}
    for Gv in (1.6, 2.0, 3.0):
        row['w2_G%.1f' % Gv] = float(fn(xv, Gv))
    tab.append(row)
    print('R/M=%6.1f  Gamma_1=%.4f   omega0^2 R^3/M: G=1.6 %+.4f  G=2 %+.4f  G=3 %+.4f'
          % (RM, row['Gamma1'], row['w2_G1.6'], row['w2_G2.0'], row['w2_G3.0']))
out['table'] = tab
# radius where Gamma = 2 becomes marginal: Gamma_1(x) = 2
xs = sp.nsolve(G1_psp - 2, x, 0.7)
RM_marg = float(2/(1 - xs**2))
print('\nGamma = 2 marginal at x = %.6f, R/M = %.6f (C = %.6f)' % (xs, RM_marg, 1/RM_marg))
out['Gamma2_marginal_R_over_M'] = RM_marg
# energy condition p <= sigma  <=> R >= 25M/12 ; Gamma_1 there:
x_ec = sp.Rational(1, 5)*1 + 0  # p=sigma <=> kappa=1 <=> sqrt f = 1/5
print('Gamma_1 at the energy-condition limit R = 25M/12:', float(g1(0.2)))
out['Gamma1_at_R_25_12'] = float(g1(0.2))
json.dump(out, open('../results/l0_potential.json', 'w'), indent=1)
