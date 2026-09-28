"""
CHECK 08 -- limits and special claims, with the independent solver:
 (a) trapped modes of ultracompact shells (Table trapped.tex, odd l=2)
 (b) approach of R(w) to the BH reflectivity as R -> 2M (Table bhlimit.tex), including R = 2.0002M
 (c) low-frequency limit R -> 1 (odd, even-C, even-A) at wM = 1e-3
 (d) closed (physical) system: |S| = |B_out/B_in| = 1 for the regular interior
 (e) the l=4, wM=0.0057 NaN point of scattering.json
"""
import json
import numpy as np
import ind_solver as I
from scipy.integrate import solve_ivp

V = json.load(open('../../First round/results/validation.json'))
S = json.load(open('../../First round/results/scattering.json'))
out = {}

print('(a) trapped modes, odd l=2: first-round exact  ->  mine')
for row in V['trapped_wkb']:
    w1 = complex(*row['exact'])
    try:
        wm = I.qnm(w1, 2, 'odd', row['R'])
        print('   R=%.2f  %.6f%+.3ei -> %.6f%+.3ei  |d|=%.1e   (WKB %.6f%+.3ei: Re err %.1f%%, Im ratio %.2f)'
              % (row['R'], w1.real, w1.imag, wm.real, wm.imag, abs(wm - w1), row['wkb'][0], row['wkb'][1],
                 100*abs(row['wkb'][0] - w1.real)/w1.real, w1.imag/row['wkb'][1]))
        out.setdefault('trapped', []).append(dict(R=row['R'], first=[w1.real, w1.imag], mine=[wm.real, wm.imag]))
    except Exception as e:
        print('   R=%.2f failed: %s' % (row['R'], e))

print('\n(b) R(w) vs BH reflectivity as R -> 2M (odd, l=2)')
bh = {round(r['w'], 3): r['R_BH'] for r in V['schwarzschild_limit']}
for w in (0.3, 0.37, 0.45, 0.6):
    line = '   w=%.2f  R_BH=%.5f :' % (w, bh[round(w, 3)])
    for Rs in (2.2, 2.05, 2.01, 2.002, 2.0002):
        Rm, Tm = I.RT(w, 2, 'odd', Rs)
        line += '  R(%.4f)=%.5f' % (Rs, Rm)
    print(line)
for w in (0.3, 0.45, 0.6):
    row = [r for r in V['schwarzschild_limit'] if abs(r['w'] - w) < 1e-9][0]
    print('   first round at w=%.2f: ' % w, {k: round(v, 5) for k, v in row.items() if k.startswith('odd')})

print('\n(c) low frequency, wM = 1e-3, l = 2:')
for par, mod in (('odd', 'C'), ('even', 'C'), ('even', 'A')):
    for R in (3.0, 6.0):
        Rm, Tm = I.RT(1e-3, 2, par, R, mod, rfar=2e5)
        print('   %s-%s R=%g: 1-R = %.2e, T = %.2e' % (par, mod, R, 1 - Rm, Tm))


print('\n(d) closed system (regular interior): |S| - 1')
for par, mod in (('odd', 'C'), ('even', 'C'), ('even', 'A')):
    worst = 0
    for w in (0.05, 0.2, 0.5, 1.0, 1.7):
        y = I.transfer(par, w, 2, 3.0, mod) @ I.interior(2, w, 3.0, 'regular')
        rfar = max(400.0, 120/w)
        y = I.integrate(y, 3.0, rfar, 2, par, w, atol=1e-14*np.abs(y).max())
        Pu, dPu = I.up_series(rfar, 2, par, w)
        Bin, Bout = np.linalg.solve(np.array([[np.conj(Pu), Pu], [np.conj(dPu), dPu]]), y)
        worst = max(worst, abs(abs(Bout/Bin) - 1))
    print('   %s-%s R=3: max_w ||S|-1| = %.1e' % (par, mod, worst))

print('\n(e) NaN/None entries in first-round scattering.json:')
for c in S['curves']:
    bad = [c['w'][i] for i, v in enumerate(c['R1']) if v is None or not np.isfinite(v)]
    if bad:
        print('   %s %s l=%d R=%g: method-1 NaN at w =' % (c['parity'], c['model'], c['l'], c['R']), bad)
with open('out_c08.json', 'w') as fh:
    json.dump(out, fh, indent=1)
