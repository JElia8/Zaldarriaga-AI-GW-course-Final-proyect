"""
Readable form of the even-parity junction equations for the static shell (model C),
in terms of the RW-gauge bulk values at r = R on each side, BEFORE substituting the
Zerilli-Moncrief master function.  Used for the derivation document and for the
comparison with Pani et al. (2009) App. A (Eqs. A20, A36-A41).
Output: even_components.txt (LaTeX strings).
"""
import sympy as sp
import derive_junction as dj

out = []
for side in ('+', '-'):
    g = dj.build_side(side, 'even', False)
    ch = dj.harmonic_components(g['h1'], 'even')
    cK = dj.harmonic_components(g['K1'], 'even')
    cE = dj.E_components(ch, cK, g['h0'], g['K0'], 'even')
    K0 = g['K0'].applyfunc(sp.simplify)
    out.append('%%%% side %s  background K_ab = %s' % (side, sp.latex(K0)))
    for nm, dct in (('dh', ch), ('dK', cK), ('dE', cE)):
        for k, v in dct.items():
            vs = sp.simplify(sp.factor(sp.cancel(v)))
            out.append('%s side %s %s:  %s' % (nm, side, k, sp.latex(vs)))
            print(nm, side, k, vs, flush=True)
with open('even_components.txt', 'w') as fh:
    fh.write('\n'.join(out))
