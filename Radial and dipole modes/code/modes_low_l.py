"""
Mode conditions for l = 0 and l = 1 from the linearized Israel matrices (israel_low_l.py).

* Picks the independent rows (cont/isr: tautau, tautheta, thetatheta at one angle; the phiphi rows and the second
  angle are proportional to them for l <= 1 and are checked to add no rank).
* det = 0 gives the frequencies Omega (shell proper time); omega = sqrt(f) Omega (exterior time).
* l = 0: compared with the exact effective-potential result of l0_potential.py.
* l = 1: closed form, marginal index, Newtonian limit (compared with PSP26 Eq. 10.57: lambda_+ = (3 Gamma - 4)/2,
  lambda_- = 0 (translation)), eigenvectors (centre-of-mass / translation check).
Output: ../results/modes_low_l.json
"""
import json
import pickle
import sympy as sp

x = sp.symbols('x', positive=True)
Om = sp.symbols('Omega')
G = sp.symbols('Gamma', positive=True)
C = sp.symbols('C', positive=True)
w2 = sp.symbols('varsigma2')         # omega^2 R^3/M, exterior time
out = {}


def load(l):
    A, cols, names = pickle.load(open('../results/israel_l%d_matrix.pkl' % l, 'rb'))
    return A, cols, names


def check_rank(A, sub_idx):
    """rank of full matrix == rank of the chosen rows, for random numeric values."""
    import random
    for _ in range(3):
        v = {x: sp.Rational(random.randint(15, 95), 100), G: sp.Rational(random.randint(11, 40), 10),
             Om: sp.Rational(random.randint(1, 90), 37)}
        assert A.subs(v).rank() == A.extract(sub_idx, list(range(A.cols))).subs(v).rank()


for l in (0, 1):
    print('=' * 80)
    print('l =', l)
    A, cols, names = load(l)
    keep = [names.index(n) for n in names if n.endswith('@0') and 'phph' not in n]
    check_rank(A, keep)
    Asq = A.extract(keep, list(range(A.cols)))
    print('independent rows:', [names[i] for i in keep])
    det = sp.factor(sp.simplify(Asq.det()))
    print('det =', det)
    # factor in Omega; each non-trivial factor is a polynomial in Omega^2
    num = sp.numer(sp.together(det))
    polys = [fa for fa, _ in sp.factor_list(num, Om)[1] if fa.has(Om)]
    print('factors containing Omega:', polys)
    Mv = (1 - x**2)/2
    res = {}
    for fa in polys:
        for so in sp.solve(fa.subs(Om, sp.sqrt(w2*Mv)/x), w2):
            so = sp.factor(sp.simplify(so))
            print('   root: omega^2 R^3/M =', so, '    (omega = sqrt(f) Omega, exterior time)')
            if so != 0:
                res['root'] = so
    out['l%d' % l] = {'det': str(det), 'roots': [str(sp.solve(fa, w2)) for fa in polys]}
    if l == 0:
        exact = (4*G*x**2 - 3*x**2 - 2*x - 1)/(2*(x + 1))
        print('   difference with the exact potential result (l0_potential.py):', sp.simplify(res['root'] - exact))
        out['l0']['agrees_with_potential'] = bool(sp.simplify(res['root'] - exact) == 0)
    if l == 1:
        r1 = res['root']
        Gc = sp.solve(sp.numer(sp.together(r1)), G)
        print('   marginal Gamma (dipole):', [sp.factor(g) for g in Gc])
        ser = sp.expand(sp.series(r1.subs(x, sp.sqrt(1 - 2*C)), C, 0, 2).removeO())
        print('   small-C expansion:', ser, '   (Newtonian PSP26 10.57: (3 Gamma - 4)/2)')
        G1 = (1 + 2*x + 3*x**2)/(4*x**2)
        print('   dipole root - monopole Gamma1 comparison: dipole marginal - Gamma1 =',
              [sp.factor(sp.simplify(g - G1)) for g in Gc])
        out['l1'].update({'omega2_R3_over_M': str(r1), 'Gamma_dip': [str(sp.factor(g)) for g in Gc]})
        # zero mode: null vector at Omega = 0 -> should be a rigid translation in the interior frame
        # (interior embedding r = R + z cos(th), th' = th - (z/R) sin(th): X_m = z, b_m = z/R; we fixed b_m = 0,
        #  so after the reparametrization zeta^th = -b_m the translation reads X_m = z, b_p = -z, X_p = ?)
        N0 = Asq.subs(Om, 0).nullspace()
        print('   null space at Omega = 0 (', cols, '):', [list(sp.simplify(v.T)) for v in N0])
        # the oscillating mode's eigenvector
        Om1 = sp.sqrt(r1*Mv)/x
        N1 = Asq.subs(Om, Om1).applyfunc(sp.simplify).nullspace()
        vec = sp.simplify(N1[0]/N1[0][0])
        print('   eigenvector of the oscillating mode (normalized X_m = 1):', list(vec.T))
        out['l1']['eigvec'] = [str(v) for v in vec]
json.dump(out, open('../results/modes_low_l.json', 'w'), indent=1)
