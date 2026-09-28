"""
Gravitational-wave perturbations of a spherical thin shell
(flat interior / Schwarzschild exterior): scattering coefficients and QNMs.

SHELL MODELS (see derivation/derivation.pdf, Sec. 1):
  * Model C  -- static perfect-fluid shell, sigma and p from Poisson (2004)
               Eqs. (3.80)-(3.81), i.e. Ipser & Sikivie (1984) Eq. (3.7a) with
               tau/sigma = -kappa(R), kappa = (1-sqrt f)/(4 sqrt f) fixed by R/M.
               TRULY STATIC background, but (as found by Pitre, Schneider &
               Poisson 2026 and reproduced here) LINEARLY UNSTABLE: an even-parity
               matter mode with purely imaginary frequency w ~ 0.6 i (M/R^3)^{1/2}.
               Perturbation EOS: polytropic, delta p = Gamma p/(sigma+p) delta sigma,
               fiducial Gamma = 2.
  * Model A  -- pure domain wall (tau = sigma), ADIABATICALLY FROZEN at the turning
               point Rdot = 0 (Rddot from I&S Eq. 3.7a). Valid for w >> w_dyn; the
               violation of the Israel constraints scales as w_dyn/w and the energy
               non-conservation as (w_dyn/w)^2 (quantified by the code).
  * Option B (static dust shell) of the task statement does NOT exist: I&S (3.7a)
               with tau = 0 gives Rddot < 0 (see shellgw/background.py).
  Odd parity: junction conditions are identical for all models (derived).
REGIME: results for real w describe the frozen/static background and are physically
  meaningful for w well above the instability / collapse rates
  (~0.6 (M/R^3)^{1/2} for model C, w_dyn for model A).

Usage:  python main.py [validate] [scatter] [qnm] [tracks] [figures] [csv]   (default: all)
Units: G = c = M = 1 (frequencies are w M).
"""
import sys
import time
import numpy as np
from multiprocessing import Pool

from shellgw import studies as S


def wgrid_main():
    return np.unique(np.concatenate([np.geomspace(0.005, 0.1, 25), np.linspace(0.1, 2.0, 191)]))


def stage_validate():
    """Validation checks required by the task (results/validation.json)."""
    from shellgw.blackhole import leaver_qnm, bh_reflectivity
    from shellgw.junction import T_even, T_odd, constraint_residual
    from shellgw.scattering import barrier_RT, closed_S
    from shellgw.qnm import find_qnm
    from shellgw.wkb import trapped_modes_bs
    from shellgw.background import (sigma, pressure_static, kappa_static, static_check_IS37a,
                                    mass_turning_point, Rddot_domain_wall, dynamical_frequency)
    import shellgw.junction_odd_C as oddC
    import shellgw.junction_odd_A as oddA
    V = {}
    # 1. background
    bg = []
    for R in (2.2, 3.0, 6.0, 10.0):
        bg.append(dict(R=R, sigma=sigma(R), p=pressure_static(R), kappa=kappa_static(R),
                       IS37a_residual_static=static_check_IS37a(R, -kappa_static(R)),
                       IS39_mass=mass_turning_point(sigma(R), R), Rddot_wall=Rddot_domain_wall(R),
                       w_dyn_wall=dynamical_frequency(R),
                       growth_rate_scale=float(np.sqrt(1 / R**3))))
    V['background'] = bg
    # 2. Schwarzschild QNMs + WKB
    V['bh_qnm'] = S.bh_qnms()
    # 3. BH reflectivity vs Vishveshwara (1970) Fig. 2 (k^2 = (2Mw)^2)
    V['vishveshwara'] = [dict(k2=(2 * w)**2, R_odd=abs(bh_reflectivity(w, 2, 'odd')[0])**2,
                              R_even=abs(bh_reflectivity(w, 2, 'even')[0])**2)
                         for w in (0.1, 0.25, 0.3535534, 0.5, 0.7071068, 0.8660254, 1.0)]
    # 4. junction limits
    V['junction'] = dict(
        odd_generated_minus_analytic=float(np.abs(oddC.T(0.3 - 0.1j, 6, 3.7, 1.0, 0.0) - T_odd(3.7)).max()),
        odd_modelA_minus_modelC=float(np.abs(oddA.T(0.3 - 0.1j, 6, 3.7, 1.0, 0.0) - oddC.T(0.3 - 0.1j, 6, 3.7, 1.0, 0.0)).max()),
        even_C_transparent=float(np.abs(T_even(0.5, 2, 5.0, 1e-8, 'C', 0.3) - np.eye(2)).max()),
        even_A_transparent=float(np.abs(T_even(0.5, 2, 5.0, 1e-8, 'A') - np.eye(2)).max()),
        det_T_even_C_times_sqrtf=[float(abs(np.linalg.det(T_even(w, 2, R, 1.0, 'C', S.vs2_of(R))) * np.sqrt(1 - 2 / R)))
                                  for R in (3.0, 6.0) for w in (0.2, 0.7)],
        modelA_constraint=[dict(R=R, w=w, constr=constraint_residual(w, 2, R, 1.0, 'A'))
                           for R in (3.0, 6.0) for w in (0.3, 1, 3, 10, 30, 100)])
    # 5. low-frequency and R -> 2M limits
    lowf = []
    for par, model in (('odd', 'C'), ('even', 'C'), ('even', 'A')):
        for R in (3.0, 6.0):
            vs2 = S.vs2_of(R) if model == 'C' else 0
            for w in (1e-3, 3e-3, 1e-2):
                Rc, Tc = barrier_RT(w, 2, par, R, 1.0, model, vs2, 1)
                lowf.append(dict(cfg=S.cfg_label(par, model), R=R, w=w, Rcoef=Rc, Tcoef=Tc))
    V['low_frequency'] = lowf
    bhl = []
    for w in (0.2, 0.3, 0.37, 0.45, 0.6):
        Rbh = abs(bh_reflectivity(w, 2, 'odd')[0])**2
        row = dict(w=w, R_BH=Rbh)
        for Rs in (2.2, 2.05, 2.01, 2.002):
            row['odd_R%.3f' % Rs] = barrier_RT(w, 2, 'odd', Rs, 1.0, 'C', 0, 1)[0]
            row['evenC_R%.3f' % Rs] = barrier_RT(w, 2, 'even', Rs, 1.0, 'C', S.vs2_of(Rs), 1)[0]
        bhl.append(row)
    V['schwarzschild_limit'] = bhl
    # 6. Pitre, Schneider & Poisson (2026) matter modes (l = 2, Gamma = 2)
    pit = []
    sR, sI = S.pn_matter_seeds(2, 2.0)
    for MR in (0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01):
        R = 1.0 / MR
        vs2 = S.vs2_of(R)
        sc = np.sqrt(1 / R**3)
        wu, ru, _ = find_qnm(1j * abs(sI) * sc, 2, 'even', R, 1.0, 'C', vs2, 1)
        wu2, _, _ = find_qnm(wu, 2, 'even', R, 1.0, 'C', vs2, 2)
        ws, rs, _ = find_qnm(sR.real * sc - 1e-4j * sc, 2, 'even', R, 1.0, 'C', vs2, 1)
        ws2, _, _ = find_qnm(ws, 2, 'even', R, 1.0, 'C', vs2, 2)
        pit.append(dict(M_over_R=MR, unstable=S.cplx(wu / sc), unstable_m2=S.cplx(wu2 / sc),
                        stable=S.cplx(ws / sc), stable_m2=S.cplx(ws2 / sc), res=[ru, rs]))
    V['pitre2026_matter'] = dict(pn_varsigma0_real=float(sR.real), pn_varsigma0_imag=float(abs(sI)), rows=pit)
    # 7. trapped-mode WKB (Bohr-Sommerfeld + Gamow) vs exact, odd l = 2
    tw = []
    for R in (2.01, 2.02, 2.05, 2.1):
        for g in trapped_modes_bs(2, 'odd', R, nmax=6):
            w, res, ok = find_qnm(g, 2, 'odd', R, 1.0, 'C', 0, 1)
            if not (ok and res < 1e-8 and abs(w - g) < 0.2 * abs(g)):
                # retry from a slightly shifted guess
                w, res, ok = find_qnm(g * 1.05 - 1e-5j, 2, 'odd', R, 1.0, 'C', 0, 1)
            tw.append(dict(R=R, wkb=S.cplx(g), exact=S.cplx(w), res=res))
    V['trapped_wkb'] = tw
    S._save('validation.json', V)
    return V


def stage_scatter(pool):
    wg = wgrid_main()
    jobs = []
    for par, model in (('odd', 'C'), ('even', 'C'), ('even', 'A')):
        for R in (2.2, 3.0, 6.0, 10.0):
            jobs.append((par, model, 2, R, S.GAMMA_FID, wg, True))
        for l in (3, 4):
            for R in (3.0, 6.0):
                jobs.append((par, model, l, R, S.GAMMA_FID, wg, False))
    for G in (1.6, 3.0):
        jobs.append(('even', 'C', 2, 3.0, G, wg, False))
    for par in ('odd', 'even'):
        for R in (2.05, 2.01):
            jobs.append((par, 'C', 2, R, S.GAMMA_FID, wg, False))
    res = pool.map(S.rt_curve, jobs, chunksize=1)
    bh = pool.map(S.bh_curve, [(l, p, wg) for l in (2, 3, 4) for p in ('odd', 'even')])
    S._save('scattering.json', dict(curves=res, bh=bh))


def stage_qnm(pool):
    jobs = [(par, model, l, R, S.GAMMA_FID)
            for l in (2, 3, 4)
            for par, model in (('odd', 'C'), ('even', 'C'), ('even', 'A'))
            for R in (2.2, 2.5, 3.0, 4.0, 6.0, 10.0)]
    res = pool.map(S.qnm_catalogue, jobs, chunksize=1)
    S._save('qnm_catalogue.json', res)


def stage_tracks(pool):
    cat = S._load('qnm_catalogue.json')
    jobs = []
    down = list(np.geomspace(6.0, 2.03, 70))[1:]
    up = list(np.geomspace(6.0, 40.0, 50))[1:]
    for c in cat:
        if c['l'] != 2 or c['R'] != 6.0 or c['model'] == 'A':
            continue
        waves = sorted([m for m in c['modes'] if m['kind'] == 'wave'], key=lambda m: m['w1'][0])[:5]
        others = [m for m in c['modes'] if m['kind'].startswith('matter')]
        for m in waves + others:
            w0 = complex(*m['w1'])
            for grid in (down, up):
                jobs.append((c['parity'], c['model'], 2, S.GAMMA_FID, 6.0, w0, grid))
    res = pool.map(S.track, jobs, chunksize=1)
    S._save('qnm_tracks.json', res)


def stage_csv():
    """QNM table: l, n, parity, model, R/M, Re, Im, method, residual, literature."""
    import csv
    import os
    cat = S._load('qnm_catalogue.json') or []
    V = S._load('validation.json') or {}
    rows = []
    # Berti, Cardoso & Starinets, CQG 26, 163001 (2009), Table (6 digits); Leaver (1985)
    lit_bh = {(2, 0): '0.373672-0.088962i', (2, 1): '0.346711-0.273915i', (2, 2): '0.301053-0.478277i',
              (2, 3): '0.251505-0.705148i', (3, 0): '0.599443-0.092703i', (3, 1): '0.582644-0.281298i',
              (3, 2): '0.551685-0.479093i', (3, 3): '0.511962-0.690337i', (4, 0): '0.809178-0.094164i',
              (4, 1): '0.796632-0.284334i', (4, 2): '0.772710-0.479908i', (4, 3): '0.739837-0.683924i'}
    lit_bh = {k: v + ' (Berti, Cardoso & Starinets 2009; Leaver 1985)' for k, v in lit_bh.items()}
    for b in V.get('bh_qnm', []):
        for key, method in (('leaver', 'Leaver continued fraction'), ('wkb3', 'WKB 3rd order'),
                            ('wkb6', 'WKB 6th order')):
            rows.append(dict(l=b['l'], n=b['n'], parity='odd=even (isospectral)', model='Schwarzschild BH',
                             R_over_M='', Gamma='', Re_wM=b[key][0], Im_wM=b[key][1], method=method,
                             residual='', literature=lit_bh.get((b['l'], b['n']), 'n/a')))
    for c in cat:
        for m in c['modes']:
            for key, method in (('w1', 'M1 transfer matrix + continued fraction'),
                                ('w2', 'M2 shooting (complex-ray DOP853)')):
                lit = 'none found'
                if c['parity'] == 'even' and c['model'] == 'C' and m['kind'].startswith('matter'):
                    lit = 'Pitre, Schneider & Poisson 2026 (arXiv:2604.05980) Figs.1-2, Eq.(7.8)'
                elif c['model'] == 'C' or c['parity'] == 'odd':
                    lit = 'same system as Pitre et al. 2026 (figures only, Figs.3-4,11-12)'
                rows.append(dict(l=c['l'], n=m['n'], parity=c['parity'],
                                 model=('C=A (odd: model independent)' if c['parity'] == 'odd' else
                                        ('C static fluid shell' if c['model'] == 'C' else
                                         'A frozen domain wall')) + (' [%s]' % m['kind']),
                                 R_over_M=c['R'], Gamma=(c['Gamma'] if c['model'] == 'C' and c['parity'] == 'even' else ''),
                                 Re_wM=m[key][0], Im_wM=m[key][1], method=method,
                                 residual=(m['res'] if key == 'w1' else m['diff']), literature=lit))
    for r_ in (V.get('pitre2026_matter', {}) or {}).get('rows', []):
        R = 1 / r_['M_over_R']
        sc = np.sqrt(1 / R**3)
        for key, kind in (('unstable', 'matter-unstable'), ('stable', 'matter-stable')):
            rows.append(dict(l=2, n=0, parity='even', model='C static fluid shell [%s]' % kind, R_over_M=R,
                             Gamma=2.0, Re_wM=r_[key][0] * sc, Im_wM=r_[key][1] * sc,
                             method='M1 transfer matrix + continued fraction', residual='',
                             literature='Pitre et al. 2026 Fig.1/2; PN limit w(R^3/M)^1/2 -> %s'
                             % ('0.5147i' if kind == 'matter-unstable' else '1.505')))
    path = os.path.join(S.RESDIR, 'qnm_table.csv')
    with open(path, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r_ in rows:
            wr.writerow(r_)
    print('wrote', path, len(rows), 'rows')


if __name__ == '__main__':
    stages = sys.argv[1:] or ['validate', 'scatter', 'qnm', 'tracks', 'csv', 'figures']
    t0 = time.time()
    with Pool(10) as pool:
        for st in stages:
            t1 = time.time()
            print('=== stage', st, flush=True)
            if st == 'validate':
                stage_validate()
            elif st == 'scatter':
                stage_scatter(pool)
            elif st == 'qnm':
                stage_qnm(pool)
            elif st == 'tracks':
                stage_tracks(pool)
            elif st == 'csv':
                stage_csv()
            elif st == 'figures':
                import make_figures
                make_figures.main()
            print('    done in %.0f s' % (time.time() - t1), flush=True)
    print('total %.0f s' % (time.time() - t0))
