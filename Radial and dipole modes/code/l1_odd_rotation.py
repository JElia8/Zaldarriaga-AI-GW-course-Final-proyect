"""
l = 1, odd parity: no oscillation, only a stationary rigid rotation; frame dragging of the interior.

Odd sector, any l (first-round derivation, odd part of D_b S^ab = 0):  d/dtau (u_A) = 0,
so for omega != 0 the shell's matter does not move (U = 0). No odd matter mode exists at any l.
The only odd l = 1 solution is stationary (omega = 0): the shell rotates slowly.

Stationary solution (metric perturbation only in g_{t phi}):
  exterior (linearized Kerr):  g_{t phi} = -(2J/r) sin^2 th           (not gauge: J is physical)
  interior (flat, rotating axes): g_{T phi} = -W_in r^2 sin^2 th       (pure gauge in flat space)
  fluid: u^phi = Omega_s u^t  (angular velocity measured in exterior time t)
Israel conditions (tau-A components; first-round Eq. for odd parity, stationary limit):
  continuity of h_{tau phi}:   H = h_t^+/sqrt f = h_T^-
  [dK_{tau A}] = -4 pi sigma H + 8 pi (sigma + p) U,   dK_{tau A} = (1/2) d_r h_t (proper-time normalized)
with U the coefficient of u_A = U X_A (X_phi = sin^2 th for l = 1).
We solve for W_in and Omega_s in terms of J and check two classic limits:
  weak field:  Omega_drag/Omega_s -> 4M/(3R)          (Thirring 1918, Lense-Thirring)
  R -> 2M:     Omega_drag/Omega_s -> 1               (complete dragging; Brill & Cohen 1966)
"""
import sympy as sp

R, J, C = sp.symbols('R J C', positive=True)
x = sp.symbols('x', positive=True)          # sqrt f(R)
r = sp.symbols('r', positive=True)
Win, U = sp.symbols('W_in U')
M = R*(1 - x**2)/2
sig = (1 - x)/(4*sp.pi*R)
p = (1 - x)/(4*x)*sig

ht_ext = -2*J/r                               # coefficient of X_phi = sin^2 th
ht_int = -Win*r**2
H_ext = (ht_ext/x).subs(r, R)                 # proper-time normalization outside: e_tau = f^{-1/2} d_t
H_int = ht_int.subs(r, R)
Win_sol = sp.solve(sp.Eq(H_ext, H_int), Win)[0]
dK_ext = (sp.diff(ht_ext, r)/2).subs(r, R)    # (sqrt f/2) d_r h_t, times f^{-1/2} from e_tau
dK_int = (sp.diff(ht_int, r)/2).subs(r, R).subs(Win, Win_sol)
H = H_int.subs(Win, Win_sol)
U_sol = sp.solve(sp.Eq(dK_ext - dK_int, -4*sp.pi*sig*H + 8*sp.pi*(sig + p)*U), U)[0]
print('interior rotation rate (interior time T):  W_in =', sp.simplify(Win_sol))
drag = sp.simplify(Win_sol*x)                 # dT/dt = sqrt f: rate in exterior time
print('dragging rate of the interior in exterior time: Omega_drag =', drag, '  (= 2J/R^3 = omega_LT(R))')
# angular momentum check: J = (sigma+p) U * integral of R^2 sin^2 th dOmega... = (8 pi/3) R^2 (sigma+p) U
print('J from the shell stress:  (8 pi/3) R^2 (sigma+p) U / J =', sp.simplify(8*sp.pi/3*R**2*(sig + p)*U_sol/J))
# u_phi = g_phiphi u^phi + g_tphi u^t = R^2 sin^2 (Omega_s - omega_LT) u^t,  u^t = 1/sqrt f
Om_s = sp.simplify(drag + x*U_sol/R**2)
ratio = sp.factor(sp.simplify(drag/Om_s))
print('Omega_drag/Omega_shell =', ratio)
print('   weak field (C -> 0):', sp.series(ratio.subs(x, sp.sqrt(1 - 2*C)), C, 0, 3))
print('   R -> 2M (x -> 0):   ', sp.limit(ratio, x, 0))
print('   at R = 3M, 6M:', [float(ratio.subs(x, sp.sqrt(1 - sp.Rational(2, k)))) for k in (3, 6)])
