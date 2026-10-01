"""
Newtonian check: the from-scratch Newtonian equations of motion of a self-gravitating fluid shell
(same equations as Full material/Second round/checks/c05_newtonian_shell.py, general l), evaluated at l = 0, 1,
compared with the M/R -> 0 limit of the relativistic l = 0, 1 frequencies found here.
For l = 0 the tangential displacement y does not exist (grad Y = 0), so only the radial equation is kept.
"""
import sympy as sp

l, G, C = sp.symbols('ell Gamma C', positive=True)
x, y = sp.symbols('x y')


def newton_matrix(lv):
    lam = lv*(lv + 1)
    rad = (-x + lam*(y + 2*x)/(2*(2*lv + 1)) + G/2*(2*x - lam*y) + sp.Rational(2 - lam, 4)*x - (2*x - lam*y)/2)
    tan = (-G/4*(2*x - lam*y) - (lam*y - (lv + 1)*x)/(2*lv + 1))
    rad, tan = sp.expand(rad), sp.expand(tan)
    if lv == 0:
        return sp.Matrix([[rad.coeff(x)]])
    return sp.Matrix([[rad.coeff(x), rad.coeff(y)], [tan.coeff(x), tan.coeff(y)]])


xs = sp.sqrt(1 - 2*C)
rel = {0: (4*G*xs**2 - 3*xs**2 - 2*xs - 1)/(2*(xs + 1)),
       1: (12*G*xs**2 - 9*xs**2 - 6*xs - 1)/(2*(3*xs + 1))}
for lv in (0, 1):
    ev = [sp.factor(e) for e in newton_matrix(lv).eigenvals()]
    lim = sp.factor(sp.limit(rel[lv], C, 0))
    print('l=%d  Newtonian varsigma^2 = %s    relativistic, C -> 0: %s' % (lv, ev, lim))
