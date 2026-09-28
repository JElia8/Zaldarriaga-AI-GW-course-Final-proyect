"""
(1) Series representation of the purely outgoing Regge-Wheeler solution about an
    arbitrary exterior point r = R2 (Leins, Nollert & Soffel 1993; Pani et al.
    2009, App. B, Eqs. B3-B7), re-derived for the e^{-i w t} convention:

        Psi(r) = chi(r) phi(z),  z = 1 - R2/r,  chi = (r-2M)^{2iMw} e^{i w r}.

    Using f chi'/chi = i w exactly, the RW equation f(f Psi')' + (w^2 - V)Psi = 0
    becomes   f phi'' + (2 i w + f') phi' - (lam/r^2 - 6M/r^3) phi = 0   (r-derivs),
    i.e. with u = 1 - z, m = M/R2:
        c(z) = u^2 (1 - 2 m u),  d(z) = -2u + 6 m u^2 + 2 i w R2,  e(z) = 6 m u - lam,
        c phi_zz + d phi_z + e phi = 0.
    This equals Pani et al. (2009) Eq. (B4) with w -> -w (their e^{+iwt}).

(2) Chandrasekhar transformation RW -> Zerilli (Chandrasekhar 1983 Sec. 4.26;
    Pani et al. 2009 Sec. III B), MP05 normalisation:
        Z = [mu(mu+2) + 72 M^2 f / (r (mu r + 6M))] Psi + 12 M dPsi/dr*.
"""
import sympy as sp

r, z, M, w, R2 = sp.symbols('r z M omega R2')
lam = sp.Symbol('lambda')
f = 1 - 2 * M / r
V = f * (lam / r**2 - 6 * M / r**3)
phi = sp.Function('phi')(r)

# ---------- (1) phi equation in r
chi = (r - 2 * M)**(2 * sp.I * M * w) * sp.exp(sp.I * w * r)
Psi = chi * phi
ode = f * sp.diff(f * sp.diff(Psi, r), r) + (w**2 - V) * Psi
claim = f * (f * sp.diff(phi, r, 2) + (2 * sp.I * w + sp.diff(f, r)) * sp.diff(phi, r)
             - (lam / r**2 - 6 * M / r**3) * phi)
print('(1) phi-equation check:', sp.simplify(sp.expand(ode / chi - claim)))
# change of variable r -> z: check c,d,e
u = 1 - z
m = M / R2
Phi = sp.Function('Phi')
rz = R2 / u
# d/dr = (u^2/R2) d/dz
p0, p1, p2 = sp.symbols('p0 p1 p2')
dphi = u**2 / R2 * p1
d2phi = u**4 / R2**2 * p2 - 2 * u**3 / R2**2 * p1
fz = (1 - 2 * M / rz)
fpz = 2 * M / rz**2
lhs = fz * d2phi + (2 * sp.I * w + fpz) * dphi - (lam / rz**2 - 6 * M / rz**3) * p0
cz = u**2 * (1 - 2 * m * u)
dz = -2 * u + 6 * m * u**2 + 2 * sp.I * w * R2
ez = 6 * m * u - lam
print('(1) z-coefficient check:', sp.simplify(lhs * R2**2 / u**2 - (cz * p2 + dz * p1 + ez * p0)))
for nm, e in (('c', cz), ('d', dz), ('e', ez)):
    print('   ', nm, sp.Poly(sp.expand(e), z).all_coeffs()[::-1])

# ---------- (2) Chandrasekhar transformation
mu = lam - 2
Lam = mu + 6 * M / r
VZ = f / Lam**2 * (mu**2 * ((mu + 2) / r**2 + 6 * M / r**3) + 36 * M**2 / r**4 * (mu + 2 * M / r))
P = sp.Function('P')(r)
P2 = ((V - w**2) * P / f - sp.diff(f, r) * sp.diff(P, r)) / f   # RW eq.
kap = mu * (mu + 2)
Zc = (kap + 72 * M**2 * f / (r * (mu * r + 6 * M))) * P + 12 * M * f * sp.diff(P, r)
res = f * sp.diff(f * sp.diff(Zc, r), r) + (w**2 - VZ) * Zc
res = res.subs(sp.Derivative(P, (r, 3)), sp.diff(P2, r)).subs(sp.Derivative(P, (r, 2)), P2)
res = res.subs(sp.Derivative(P, (r, 2)), P2)
print('(2) Chandrasekhar map RW->Zerilli residual:', sp.simplify(res))
