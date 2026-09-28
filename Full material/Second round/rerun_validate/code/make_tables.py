"""Generate LaTeX tables for the derivation document from results/*.json."""
import os
import json
import numpy as np
from shellgw import studies as S

TDIR = os.path.join(os.path.dirname(__file__), '..', 'derivation', 'tables')


def w(name, s):
    os.makedirs(TDIR, exist_ok=True)
    with open(os.path.join(TDIR, name), 'w', encoding='utf8') as fh:
        fh.write(s)


def cfmt(z, d=6):
    return '$%.*f%+.*fi$' % (d, z[0], d, z[1])


def main():
    V = S._load('validation.json')
    cat = S._load('qnm_catalogue.json')
    data = S._load('scattering.json')
    # ---------------- background
    rows = []
    for b in V['background']:
        rows.append('%.1f & %.5f & %.3e & %.4f & %.4f & %.2e & %.4f & %.4f & %.4f \\\\' % (
            b['R'], b['sigma'], b['p'], b['kappa'], S.vs2_of(b['R']), abs(b['IS37a_residual_static']),
            b['Rddot_wall'], b['w_dyn_wall'], b['growth_rate_scale']))
    w('background.tex', '\\begin{tabular}{rrrrrrrrr}\\toprule\n'
      '$R/M$ & $\\sigma M$ & $pM$ & $\\kappa=p/\\sigma$ & $v_s^2(\\Gamma{=}2)$ & I\\&S (3.7a) resid. & '
      '$\\ddot R_{\\rm wall}M$ & $\\omega_{\\rm dyn}M$ & $(M/R^3)^{1/2}M$\\\\\\midrule\n' + '\n'.join(rows) +
      '\n\\bottomrule\\end{tabular}\n')
    # ---------------- BH QNMs + WKB
    rows = []
    for b in V['bh_qnm']:
        rows.append('%d & %d & %s & %s & %s & %s \\\\' % (b['l'], b['n'], cfmt(b['leaver'], 8), cfmt(b['wkb1'], 4),
                                                         cfmt(b['wkb3'], 5), cfmt(b['wkb6'], 6)))
    w('bh_qnm.tex', '\\begin{tabular}{rrllll}\\toprule\n$\\ell$ & $n$ & Leaver CF (this code) & WKB 1st & WKB 3rd & '
      'WKB 6th\\\\\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')
    # ---------------- Pitre comparison
    P = V['pitre2026_matter']
    rows = []
    for r_ in P['rows']:
        rows.append('%.2f & $%.6f\\,i$ & %.1e & $%.6f%+.2e\\,i$ & %.1e \\\\' % (
            r_['M_over_R'], r_['unstable'][1], abs(complex(*r_['unstable']) - complex(*r_['unstable_m2'])),
            r_['stable'][0], r_['stable'][1], abs(complex(*r_['stable']) - complex(*r_['stable_m2']))))
    rows.append('$\\to0$ (PN, PSP26 Eq.~7.8) & $%.6f\\,i$ & & $%.6f$ & \\\\' % (P['pn_varsigma0_imag'], P['pn_varsigma0_real']))
    w('pitre.tex', '\\begin{tabular}{rllll}\\toprule\n$M/R$ & unstable: $\\omega(R^3/M)^{1/2}$ & $|\\Delta_{12}|$ & '
      'stable pair: $\\omega(R^3/M)^{1/2}$ & $|\\Delta_{12}|$\\\\\\midrule\n' + '\n'.join(rows) +
      '\n\\bottomrule\\end{tabular}\n')
    # ---------------- validation summary
    J = V['junction']
    lines = []
    lines.append('Odd: generated vs analytic $\\mathsf T$ & %.1e \\\\' % J['odd_generated_minus_analytic'])
    lines.append('Odd: $\\mathsf T^{\\rm A}-\\mathsf T^{\\rm C}$ & %.1e \\\\' % J['odd_modelA_minus_modelC'])
    lines.append('Even C: $|\\mathsf T-\\mathbb{I}|$ at $M=10^{-8}$ (transparent limit) & %.1e \\\\' % J['even_C_transparent'])
    lines.append('Even A: $|\\mathsf T-\\mathbb{I}|$ at $M=10^{-8}$ & %.1e \\\\' % J['even_A_transparent'])
    lines.append('Even C: $\\max|\\det\\mathsf T\\,\\sqrt{f_R}-1|$ & %.1e \\\\' % max(abs(x - 1) for x in J['det_T_even_C_times_sqrtf']))
    for r_ in J['modelA_constraint']:
        if r_['w'] in (1, 10, 100):
            lines.append('Even A: constraint violation, $R=%gM$, $\\omega M=%g$ & %.1e \\\\' % (r_['R'], r_['w'], r_['constr']))
    if data:
        for key, (par, model) in (('odd', ('odd', 'C')), ('even-C', ('even', 'C')), ('even-A', ('even', 'A'))):
            for R in (3.0, 6.0):
                c = [c for c in data['curves'] if c['l'] == 2 and c['R'] == R and c['parity'] == par and
                     c['model'] == model and abs(c['Gamma'] - 2) < 1e-9][0]
                e2 = np.max(np.abs(np.array(c['R2']) + np.array(c['T2']) - 1))
                d12 = max(np.max(np.abs(np.array(c['R1']) - np.array(c['R2']))),
                          np.max(np.abs(np.array(c['T1']) - np.array(c['T2']))))
                dbw = np.max(np.abs(np.array(c['Rbw']) - np.array(c['R2'])))
                lines.append('%s, $\\ell=2$, $R=%gM$: $\\max_\\omega|\\mathcal R+\\mathcal T-1|$ (M2) / '
                             '$\\max|\\Delta_{12}|$ / Boyanov estimator & %.1e / %.1e / %.1e \\\\' % (key, R, e2, d12, dbw))
    lowf = V['low_frequency']
    lines.append('Low frequency: $\\max|1-\\mathcal R|$ at $\\omega M=10^{-3}$ (all models) & %.1e \\\\' %
                 max(abs(1 - r_['Rcoef']) for r_ in lowf if r_['w'] == 1e-3))
    w('validation.tex', '\\begin{tabular}{lr}\\toprule\nCheck & value\\\\\\midrule\n' + '\n'.join(lines) +
      '\n\\bottomrule\\end{tabular}\n')
    # ---------------- Schwarzschild limit
    rows = []
    for r_ in V['schwarzschild_limit']:
        rows.append('%.2f & %.5f & %.5f & %.5f & %.5f & %.5f & %.5f & %.5f\\\\' % (
            r_['w'], r_['R_BH'], r_['odd_R2.200'], r_['odd_R2.050'], r_['odd_R2.002'],
            r_['evenC_R2.200'], r_['evenC_R2.050'], r_['evenC_R2.010']))
    w('bhlimit.tex', '\\begin{tabular}{rrrrrrrr}\\toprule\n$\\omega M$ & BH & odd $2.2M$ & odd $2.05M$ & odd $2.002M$ '
      '& even-C $2.2M$ & even-C $2.05M$ & even-C $2.01M$\\\\\\midrule\n' + '\n'.join(rows) +
      '\n\\bottomrule\\end{tabular}\n')
    # ---------------- shell QNM tables (l = 2)
    if cat:
        for l in (2, 3, 4):
            rows = []
            for R in (2.2, 3.0, 6.0, 10.0):
                for par, model in (('odd', 'C'), ('even', 'C'), ('even', 'A')):
                    c = [c for c in cat if c['l'] == l and c['R'] == R and c['parity'] == par and c['model'] == model]
                    if not c:
                        continue
                    ms = sorted([m for m in c[0]['modes'] if m['kind'] == 'wave'], key=lambda m: m['w1'][0])
                    ms = [m for m in ms if m['diff'] < 1e-4][:4]
                    extra = [m for m in c[0]['modes'] if m['kind'].startswith('matter')]
                    s = ', '.join('$%.5f%+.5fi$' % tuple(m['w1']) for m in ms)
                    if extra:
                        s += '; matter: ' + ', '.join('$%.5f%+.5fi$' % tuple(m['w1']) for m in extra)
                    rows.append('%g & %s & %s \\\\' % (R, S.cfg_label(par, model), s))
            w('shell_qnm_l%d.tex' % l, '\\begin{tabular}{rlp{12cm}}\\toprule\n$R/M$ & sector & lowest wave modes '
              '$\\omega M$ (ordered by ${\\rm Re}\\,\\omega$; Methods 1 and 2 agree to $<10^{-4}$)\\\\\\midrule\n' +
              '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')
    # ---------------- wave modes vs PSP26 Figs. 3-4 (values read off the figures, +-0.02)
    readings = [('odd', 'C', 6.0, 0, 1.47, -0.63), ('odd', 'C', 6.0, 1, 4.1, -0.84),
                ('even', 'C', 4.0, 0, 1.05, -0.655), ('even', 'C', 10.0, 0, 0.62, -0.825),
                ('even', 'C', 10.0, 1, 2.95, -0.91)]
    rows = []
    if cat:
        for par, model, R, n, reR, imR in readings:
            c = [c for c in cat if c['l'] == 2 and c['R'] == R and c['parity'] == par and c['model'] == model][0]
            ms = sorted([m for m in c['modes'] if m['kind'] == 'wave' and m['w1'][1] < -0.1],
                        key=lambda m: m['w1'][0])
            m = ms[n]
            xi = 0.5 * np.log(2 * 6 * R)
            rows.append('%s & %.3f & %d & %.3f & %.3f & %.2f & %.3f \\\\' % (
                par, 1 / R, n + 1, R * m['w1'][0], R * m['w1'][1] / xi, reR, imR))
        w('psp_wave.tex', '\\begin{tabular}{lrrrrrr}\\toprule\nparity & $M/R$ & mode & ${\\rm Re}(R\\omega)$ & '
          '${\\rm Im}(R\\omega)/\\xi$ & PSP26 Re & PSP26 Im$/\\xi$\\\\\\midrule\n' + '\n'.join(rows) +
          '\n\\bottomrule\\end{tabular}\n')
    # ---------------- trapped modes
    rows = []
    for t in V['trapped_wkb']:
        if t['res'] < 1e-8:
            rows.append('%.2f & $%.6f%+.3e\\,i$ & $%.6f%+.3e\\,i$ \\\\' % (t['R'], t['exact'][0], t['exact'][1],
                                                                          t['wkb'][0], t['wkb'][1]))
    w('trapped.tex', '\\begin{tabular}{rll}\\toprule\n$R/M$ & exact (M1) & WKB (Bohr--Sommerfeld + Gamow)\\\\\\midrule\n'
      + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')
    print('tables written to', TDIR)


if __name__ == '__main__':
    main()
