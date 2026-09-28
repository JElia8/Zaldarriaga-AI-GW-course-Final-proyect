"""
CHECK 05 -- Newtonian non-radial oscillations of a self-gravitating 2D fluid shell,
derived from scratch in the second round (independent of PSP26 and of the first round).

Shell: radius R, surface density sigma, 2D pressure p, mass M = 4 pi R^2 sigma (G = 1).
Equilibrium (normal force balance, gravity = average of inner (0) and outer (-M/R^2) fields):
    2p/R = sigma M/(2R^2)  ->  p = sigma M/(4R)       [= kappa sigma, kappa -> M/(4R) in GR]
Perturbation (Lagrangian displacement xi_r Y rhat + xi_h grad_S2 Y, e^{-i w t}),  x = xi_r/R, y = xi_h/R:
  continuity:      Delta sigma/sigma = -(2x - lam y)
  EOS:             Delta p = Gamma p Delta sigma/sigma
  gravity:         exterior sees Sigma_out = sigma(lam y + l x), interior Sigma_in = sigma(lam y - (l+1) x)
                   (radial displacement of the mass sheet shifts its multipole moments)
  radial eq.:      -w^2 R x = M x/R^2 + (1/2)(dg_in + dg_out) + [2 Dp/R - p (2 - lam) x/R]/sigma
                               - (2p/(R sigma)) Delta sigma/sigma          (Laplace-type membrane force)
  tangential eq.:  -w^2 R y = -Delta p/(sigma R) - dPhi_in(R)/R
Result: a 2x2 eigenproblem for s^2 = w^2 R^3/M.  Compared with PSP26 Eqs. (7.6), (7.8), (10.62).
"""
import sympy as sp

l, G, s2 = sp.symbols('ell Gamma s2')
lam = l*(l + 1)
x, y = sp.symbols('x y')
# (in units M/R^3 for w^2; everything divided by R)
rad = (-x + lam*(y + 2*x)/(2*(2*l + 1)) + G/2*(2*x - lam*y) + (2 - lam)*x/4 - (2*x - lam*y)/2)
tan = (-G/4*(2*x - lam*y) - (lam*y - (l + 1)*x)/(2*l + 1))
rad, tan = sp.expand(rad), sp.expand(tan)
Mx = sp.Matrix([[rad.coeff(x), rad.coeff(y)], [tan.coeff(x), tan.coeff(y)]])
tr = sp.simplify(Mx.trace())
det = sp.factor(sp.simplify(Mx.det()))
print('trace =', sp.factor(tr))
print('det   =', det)
# PSP26 Eq. (7.8): s0^2 = (1/8)[(L+4)G - (L+6)] +- (1/8) sqrt(...)
L = l**2 + l
A = (L + 4)*G - (L + 6)
disc78 = (L + 4)**2*G**2 + 2*(2*l**5 - 3*l**4 - 40*l**3 - 33*l**2 - 46*l - 24)/(2*l + 1)*G \
    + (2*l**5 - 11*l**4 + 60*l**3 + 53*l**2 + 52*l + 36)/(2*l + 1)
# roots of s^4 - tr s^2 + det = 0:   s^2 = tr/2 +- sqrt(tr^2/4 - det)
print('trace - A/4 (sum of the two roots of (7.8)) =', sp.simplify(tr - A/4))
print('(tr^2/4 - det) - disc(7.8)/64 =', sp.simplify(tr**2/4 - det - disc78/64))
# PSP26 Eq. (7.6) as printed: s^4 - (1/4)A s^2 - (l-1)l(l+1)(l+2)[(2l-3)G - 4]/(16(2l+1)) = 0
c76 = -(l - 1)*l*(l + 1)*(l + 2)*((2*l - 3)*G - 4)/(16*(2*l + 1))
print('det - [constant of (7.6) as printed] =', sp.factor(sp.simplify(det - c76)))
c_corr = -(l - 1)*l*(l + 1)*((l + 2)*(2*l - 3)*G - 4*(l - 2))/(16*(2*l + 1))
print('det - [corrected constant  -(l-1)l(l+1)[(l+2)(2l-3)G - 4(l-2)]/(16(2l+1))] =',
      sp.simplify(det - c_corr))
for lv, Gv in ((2, 2), (3, 2), (2, 1.6), (2, 3)):
    ev = [complex(sp.N(e)) for e in Mx.subs({l: lv, G: Gv}).eigenvals()]
    print('  l=%d Gamma=%.1f: s^2 =' % (lv, Gv), ['%.6f' % e.real for e in ev],
          ' -> s =', ['%.6f%+.6fi' % (complex(sp.sqrt(e)).real, complex(sp.sqrt(e)).imag) for e in ev])
