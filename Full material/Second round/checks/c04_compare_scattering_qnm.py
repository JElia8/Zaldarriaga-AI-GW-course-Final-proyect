"""
CHECK 04 -- independent recomputation of R(w), T(w) and of QNMs with ind_solver.py,
compared with the first-round results (scattering.json, qnm_catalogue.json, validation.json).
"""
import json
import time
import numpy as np
import ind_solver as I

FR = '../../First round/results/'
S = json.load(open(FR + 'scattering.json'))
CAT = json.load(open(FR + 'qnm_catalogue.json'))
V = json.load(open(FR + 'validation.json'))
out = dict(RT=[], qnm=[], pitre=[])

# ------------------------------------------------------------------ R, T
sel = [('odd', 'C', 2, 3.0), ('odd', 'C', 2, 6.0), ('odd', 'C', 3, 3.0), ('odd', 'C', 2, 2.2),
       ('even', 'C', 2, 3.0), ('even', 'C', 2, 6.0), ('even', 'C', 4, 6.0), ('even', 'A', 2, 3.0),
       ('even', 'A', 2, 6.0)]
print('R,T: |R_mine - R_first|, |T_mine - T_first|, |R+T-1|_mine  (max over sampled w)')
for par, mod, l, R in sel:
    c = [c for c in S['curves'] if c['parity'] == par and c['model'] == mod and c['l'] == l and c['R'] == R
         and abs(c['Gamma'] - 2) < 1e-9][0]
    idx = np.unique(np.linspace(0, len(c['w']) - 1, 12).astype(int))
    dR = dT = dE = 0.0
    t0 = time.time()
    for i in idx:
        w = c['w'][i]
        if c['R1'][i] is None or not np.isfinite(c['R1'][i]):
            continue
        Rm, Tm = I.RT(w, l, par, R, mod)
        dR, dT, dE = max(dR, abs(Rm - c['R1'][i])), max(dT, abs(Tm - c['T1'][i])), max(dE, abs(Rm + Tm - 1))
        out['RT'].append(dict(cfg=[par, mod, l, R], w=w, R_mine=Rm, T_mine=Tm, R_first=c['R1'][i], T_first=c['T1'][i]))
    print('  %-4s %s l=%d R=%-4g : %.1e  %.1e  %.1e   (%.0f s)' % (par, mod, l, R, dR, dT, dE, time.time() - t0))

# high-frequency delta-barrier law for odd parity
print('\nodd high-frequency law R ~ Delta^2/(4 w^2):')
for R in (3.0, 6.0):
    s = np.sqrt(1 - 2/R)
    D = s*(1 - s)/R
    for w in (2.0, 4.0, 8.0):
        Rm, Tm = I.RT(w, 2, 'odd', R)
        print('  R=%g w=%g  R_mine=%.4e  Delta^2/(4w^2+Delta^2)=%.4e  ratio=%.4f' % (R, w, Rm, D*D/(4*w*w + D*D), Rm/(D*D/(4*w*w + D*D))))
        out.setdefault('hf', []).append(dict(R=R, w=w, Rm=Rm, delta_law=D*D/(4*w*w + D*D)))

# ------------------------------------------------------------------ QNMs
print('\nQNMs: first-round M1 value -> my root  (|diff|)')
pick = [('odd', 'C', 2, 6.0), ('odd', 'C', 2, 10.0), ('odd', 'C', 2, 3.0), ('odd', 'C', 2, 2.2), ('odd', 'C', 3, 6.0),
        ('even', 'C', 2, 6.0), ('even', 'C', 2, 10.0), ('even', 'C', 2, 3.0), ('even', 'A', 2, 6.0),
        ('even', 'C', 3, 6.0)]
for par, mod, l, R in pick:
    c = [c for c in CAT if c['parity'] == par and c['model'] == mod and c['l'] == l and c['R'] == R][0]
    modes = sorted(c['modes'], key=lambda m: (m['kind'] != 'wave', m['w1'][0]))
    waves = [m for m in modes if m['kind'] == 'wave'][:4]
    matter = [m for m in modes if m['kind'] != 'wave']
    for m in waves + matter:
        w1 = complex(*m['w1'])
        if w1.real < 0 or (m['kind'] != 'wave' and abs(w1.real) < 1e-9 and w1.imag < 0):
            pass
        try:
            wm = I.qnm(w1, l, par, R, mod)
        except Exception as e:
            print('  %s %s l=%d R=%g %-18s %.6f%+.6fi  FAILED (%s)' % (par, mod, l, R, m['kind'], w1.real, w1.imag, e))
            continue
        print('  %s %s l=%d R=%-4g %-18s %.8f%+.8fi -> %.8f%+.8fi  |d|=%.1e'
              % (par, mod, l, R, m['kind'], w1.real, w1.imag, wm.real, wm.imag, abs(wm - w1)))
        out['qnm'].append(dict(cfg=[par, mod, l, R], kind=m['kind'], first=[w1.real, w1.imag], mine=[wm.real, wm.imag]))

# ------------------------------------------------------------------ PSP26 matter modes (Table pitre)
print('\nMatter modes (even C, l=2, Gamma=2) in units (M/R^3)^1/2:')
for row in V['pitre2026_matter']['rows']:
    R = 1/row['M_over_R']
    sc = R**-1.5
    for key in ('unstable', 'stable'):
        w1 = complex(*row[key])*sc
        try:
            wm = I.qnm(w1, 2, 'even', R, 'C')
        except Exception as e:
            print('   failed', e); continue
        print('  M/R=%.2f %-8s first: %.6f%+.6fi   mine: %.6f%+.6fi   |d|/scale=%.1e'
              % (row['M_over_R'], key, (w1/sc).real, (w1/sc).imag, (wm/sc).real, (wm/sc).imag, abs(wm - w1)/sc))
        out['pitre'].append(dict(MR=row['M_over_R'], kind=key, first=[(w1/sc).real, (w1/sc).imag],
                                 mine=[(wm/sc).real, (wm/sc).imag]))
with open('out_c04.json', 'w') as fh:
    json.dump(out, fh, indent=1)
