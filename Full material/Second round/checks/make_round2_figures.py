"""Figures of the second-round verification (written to ../figures_round2, PDF + PNG)."""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures_round2')
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, '..', 'fr_code_pristine'))
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']
GRAY = '#8a8984'
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11, 'legend.fontsize': 8.5,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'lines.linewidth': 1.8, 'savefig.bbox': 'tight', 'mathtext.fontset': 'cm',
                     'axes.spines.top': True, 'axes.grid': False})


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, name + '.' + ext), dpi=220 if ext == 'png' else None)
    plt.close(fig)
    print('saved', name)


# ------------------------------------------------------------------ F1 thick shell convergence
d = json.load(open(os.path.join(HERE, 'out_c03b.json')))
keys = [k for k in d[0] if k not in ('w', 'eps', 'Phi_in_minus_half_ln_f')]
labels = ['first round: $\\Delta=4\\pi\\sigma\\sqrt{f_R}$', 'wrong: $\\Delta=0$', 'wrong: $\\Delta=4\\pi\\sigma$',
          'wrong: $\\Delta\\times2$']
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, w in zip(axs, (0.2, 0.5, 1.0)):
    rows = [r for r in d if abs(r['w'] - w) < 1e-9]
    eps = np.array([r['eps'] for r in rows])
    for i, k in enumerate(keys):
        ax.loglog(eps, [r[k] for r in rows], 'o-', color=C[i] if i == 0 else [GRAY, C[1], C[3]][i - 1],
                  ms=4, lw=2.2 if i == 0 else 1.2, label=labels[i])
    ax.loglog(eps, 0.09*eps, ':', color='k', lw=1)
    ax.text(0.0075, 0.00042, '$\\propto\\varepsilon$', fontsize=9)
    ax.set_title('$\\omega M=%.1f$, $\\ell=2$, $R=3M$' % w)
    ax.set_xlabel('shell half-thickness $\\varepsilon/M$')
axs[0].set_ylabel('mismatch thick vs thin (sin of angle)')
axs[0].legend(loc='lower right', frameon=False)
fig.suptitle('Odd-parity junction: smooth anisotropic shell ($p_r=0$) $\\to$ thin shell', y=1.02)
save(fig, 'r2_fig1_thick_shell_limit')

# ------------------------------------------------------------------ F2 matter modes: PN limit and PSP26
V = json.load(open('../../First round/results/validation.json'))
rows = V['pitre2026_matter']['rows']
MR = np.array([r['M_over_R'] for r in rows])
un = np.array([r['unstable'][1] for r in rows])
st = np.array([r['stable'][0] for r in rows])
psp_MR = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])          # read off PSP26 Figs. 1-2 (+-0.002)
psp_un = np.array([0.518, 0.535, 0.557, 0.582, 0.608, 0.637, 0.668])
psp_st = np.array([1.495, 1.454, 1.398, 1.336, 1.267, 1.188, 1.096])
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
fig.subplots_adjust(wspace=0.38)
axs[0].plot(MR, un, 'o-', color=C[1], ms=5, label='first round (full GR)')
axs[0].plot(psp_MR, psp_un, 'x', color=C[0], ms=7, mew=1.6, label='read off PSP26 Fig. 1')
axs[0].plot([0], [0.514695], '*', color='k', ms=11, label='my Newtonian derivation')
axs[0].set_xlabel('$M/R$'); axs[0].set_ylabel('${\\rm Im}\\,\\omega\\,(R^3/M)^{1/2}$ (unstable)')
axs[0].legend(frameon=False, loc='upper left')
axs[1].plot(MR, st, 's-', color=C[1], ms=5, label='first round')
axs[1].plot(psp_MR, psp_st, 'x', color=C[0], ms=7, mew=1.6, label='PSP26 Fig. 2')
axs[1].plot([0], [1.504962], '*', color='k', ms=11, label='Newtonian')
axs[1].set_xlabel('$M/R$'); axs[1].set_ylabel('${\\rm Re}\\,\\omega\\,(R^3/M)^{1/2}$ (stable pair)')
axs[1].legend(frameon=False)
Gs = np.array([1.81, 1.9, 2.0, 2.1, 2.2, 2.3, 2.46])
gun = np.array([0.6437, 0.6254, 0.6077, 0.5924, 0.5789, 0.5672, 0.5511])
gst = np.array([1.1383, 1.2002, 1.2671, 1.3318, 1.3945, 1.4552, 1.5486])
axs[2].plot(Gs, gun, 'o-', color=C[1], ms=5, label='unstable (Im), my solver')
axs[2].plot(Gs, gst, 's-', color=C[2], ms=5, label='stable (Re), my solver')
axs[2].plot([1.81, 2.46], [0.643, 0.551], 'x', color=C[0], ms=8, mew=1.6, label='PSP26 Figs. 1-2 end points')
axs[2].plot([1.81, 2.46], [1.14, 1.548], 'x', color=C[0], ms=8, mew=1.6)
axs[2].set_xlabel('adiabatic index $\\Gamma$ ($M/R=0.2$)'); axs[2].set_ylabel('$\\omega\\,(R^3/M)^{1/2}$')
axs[2].legend(frameon=False, loc='center left')
fig.suptitle('Even-parity matter modes of the static fluid shell (model C, $\\ell=2$)', y=1.02)
save(fig, 'r2_fig2_matter_modes')

# ------------------------------------------------------------------ F3 model A prescription dependence
from shellgw import junction as J
import shellgw.junction_even_A as EA
from shellgw.interior import ingoing_at_shell
from shellgw.exterior import up_at_R_cf


def RT_rows(w, R, rows, model='A'):
    if model == 'A':
        A = EA.A(w, 6, R, 1.0, 0.0); B = EA.B(w, 6, R, 1.0, 0.0)
        T = np.linalg.solve(A[rows], B[rows])[-2:]
    elif model == 'C':
        s = np.sqrt(1 - 2/R); k = (1 - s)/(4*s)
        T = J.T_even(w, 2, R, 1.0, 'C', 2*k/(1 + k))
    else:
        T = J.T_odd(R)
    par = 'odd' if model == 'odd' else 'even'
    yp = T @ ingoing_at_shell(w, 2, R)
    up = up_at_R_cf(w, 2, par, R)
    f = 1 - 2/R
    L = f*up[1]/up[0]
    up = up*np.sqrt(w/L.imag)/abs(up[0])
    Bin, Bout = np.linalg.solve(np.array([[np.conj(up[0]), up[0]], [np.conj(up[1]), up[1]]]), yp)
    return abs(Bout/Bin)**2, 1/abs(Bin)**2


choices = [([0, 1, 2, 3, 6, 7], 'trace + trace-free (first round)', C[0], 2.4),
           ([0, 1, 2, 3, 4, 7], '$\\tau\\tau$ + trace-free', C[1], 1.6),
           ([0, 1, 2, 3, 5, 7], '$\\tau A$ + trace-free', C[2], 1.6),
           ([0, 1, 2, 3, 4, 6], '$\\tau\\tau$ + trace', C[3], 1.6)]
ws = np.geomspace(0.3, 30, 60)
fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.4), sharex=True)
out3 = {}
for j, R in enumerate((3.0, 6.0)):
    for rows, lab, col, lw in choices:
        RR, TT = zip(*[RT_rows(w, R, rows) for w in ws])
        RR, TT = np.array(RR), np.array(TT)
        axs[0, j].loglog(ws, RR, color=col, lw=lw, label=lab)
        axs[1, j].loglog(ws, np.abs(RR + TT - 1) + 1e-17, color=col, lw=lw, label=lab)
        out3['%g %s' % (R, lab)] = dict(w=list(ws), R=list(RR), viol=list(np.abs(RR + TT - 1)))
    Rc = np.array([RT_rows(w, R, None, 'C')[0] for w in ws])
    Ro = np.array([RT_rows(w, R, None, 'odd')[0] for w in ws])
    axs[0, j].loglog(ws, Rc, ':', color='k', lw=1.3, label='model C (even), reference')
    axs[0, j].loglog(ws, Ro, '--', color=GRAY, lw=1.1, label='odd parity (any model)')
    axs[0, j].set_title('frozen domain wall (model A), $\\ell=2$, $R=%gM$' % R)
    axs[1, j].set_xlabel('$\\omega M$')
axs[0, 0].set_ylabel('$\\mathcal{R}(\\omega)$')
axs[1, 0].set_ylabel('$|\\mathcal{R}+\\mathcal{T}-1|$')
axs[0, 0].legend(frameon=False, fontsize=7.5, loc='lower left')
axs[1, 0].set_ylim(1e-16, 3)
axs[1, 1].set_ylim(1e-16, 3)
fig.suptitle('Model A is prescription dependent: which 2 of the 4 Israel equations are imposed '
             '(metric continuity always imposed)', y=1.0, fontsize=10.5)
save(fig, 'r2_fig3_modelA_prescriptions')
json.dump(out3, open(os.path.join(HERE, 'out_fig3.json'), 'w'))

# ------------------------------------------------------------------ F4 timescales
Rg = np.geomspace(2.1, 40, 200)
s = np.sqrt(1 - 2/Rg)
wdyn = s*np.sqrt((1 + 3*s)/(2*Rg)/Rg)
cat = json.load(open('../../First round/results/qnm_catalogue.json'))
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.loglog(Rg, wdyn, color=C[0], label='$\\omega_{\\rm dyn}$ (domain-wall collapse, model A)')
ax.loglog(Rg, 0.6*Rg**-1.5, color=C[1], label='$0.6\\,(M/R^3)^{1/2}$ (instability, model C)')
ax.loglog(Rg, 1/Rg, ':', color=GRAY, label='$1/R$')
for par, mod, mk, col in (('even', 'A', '^', C[0]), ('even', 'C', 's', C[1])):
    xs, ys = [], []
    for c in cat:
        if c['parity'] == par and c['model'] == mod and c['l'] == 2:
            wv = sorted([m for m in c['modes'] if m['kind'] == 'wave'], key=lambda m: -m['w1'][1])
            if wv:
                xs.append(c['R']); ys.append(abs(complex(*wv[0]['w1'])))
    ax.loglog(xs, ys, mk, color=col, ms=6, mfc='none', mew=1.4, label='$|\\omega|$ of least-damped wave QNM, even-%s' % mod)
ax.set_xlabel('$R/M$'); ax.set_ylabel('rate $\\times M$')
ax.minorticks_off()
ax.set_xticks([2, 3, 4, 6, 10, 20, 40]); ax.set_xticklabels(['2', '3', '4', '6', '10', '20', '40'])
ax.legend(frameon=False, fontsize=7.5)
ax.set_title('Adiabatic condition: wave QNMs vs background rates')
save(fig, 'r2_fig4_timescales')

# ------------------------------------------------------------------ F5 BH limit
Rs = np.array([2.2, 2.05, 2.01, 2.002, 2.0002])
data = {0.30: [0.92247, 0.93661, 0.94122, 0.94729, 0.94481], 0.45: [0.04894, 0.10167, 0.07208, 0.07539, 0.07343],
        0.60: [0.00554, 0.00319, 0.00205, 0.00021, 0.00059]}
bh = {0.30: 0.94521, 0.45: 0.07559, 0.60: 0.00066}
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for i, (w, vals) in enumerate(data.items()):
    ax.semilogx(Rs - 2, np.array(vals)/bh[w], 'o-', color=C[i], ms=5, label='$\\omega M=%.2f$' % w)
ax.axhline(1, color=GRAY, lw=1)
ax.set_xlabel('$(R-2M)/M$'); ax.set_ylabel('$\\mathcal{R}_{\\rm shell}/\\mathcal{R}_{\\rm BH}$ (odd, $\\ell=2$)')
ax.set_title('Approach to the black-hole reflectivity as $R\\to2M$')
ax.legend(frameon=False)
save(fig, 'r2_fig5_bh_limit')
