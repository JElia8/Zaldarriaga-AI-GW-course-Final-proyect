"""Figures for the l = 0 and l = 1 study (reads ../results, the first-round QNM catalogue; writes ../figures)."""
import os
import json
import pickle
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
CAT = os.path.join(HERE, '..', '..', 'Full material', 'First round', 'results', 'qnm_catalogue.json')
BLUE, ORANGE, AQUA, YELLOW, PINK, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7'
GRAY, INK = '#8a8984', '#222222'
plt.rcParams.update({'font.family': 'serif', 'font.size': 10.5, 'axes.labelsize': 11.5, 'legend.fontsize': 9,
                     'axes.titlesize': 11, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'lines.linewidth': 1.8,
                     'savefig.bbox': 'tight', 'mathtext.fontset': 'cm', 'legend.frameon': False})
NUM = {}


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, name + '.' + ext), dpi=230 if ext == 'png' else None)
    plt.close(fig)
    print('saved', name)


# closed forms (l0_potential.py, modes_low_l.py), x = sqrt(f(R))
def w2_l0(x, G):
    return (4*G*x**2 - 3*x**2 - 2*x - 1)/(2*(x + 1))


def w2_l1(x, G):
    return (12*G*x**2 - 9*x**2 - 6*x - 1)/(2*(3*x + 1))


def G1(x):
    return (1 + 2*x + 3*x**2)/(4*x**2)


def Gdip(x):
    return (3*x + 1)**2/(12*x**2)


def Gcaus(x):            # v_s^2 = Gamma kappa/(1+kappa) <= 1  <=>  Gamma <= (1+kappa)/kappa
    return (3*x + 1)/(1 - x)


def xof(RM):
    return np.sqrt(1 - 2/np.asarray(RM, float))


def catalogue_unstable(l):
    cat = json.load(open(CAT))
    pts = {}
    for e in cat:
        if e['parity'] != 'even' or e['model'] != 'C' or e['l'] != l or abs(e['Gamma'] - 2) > 1e-9:
            continue
        for md in e['modes']:
            w = complex(*md['w1'])
            if md['kind'].startswith('matter') and w.imag > 0 and abs(w.real) < 1e-6:
                pts[e['R']] = w.imag
    R = np.array(sorted(pts))
    return R, np.array([pts[r] for r in R])


# ------------------------------------------------------------------ F1: omega^2 for l = 0, 1 vs R/M
def fig_omega2():
    RM = np.linspace(25/12, 40, 2000)
    x = xof(RM)
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    for G, ls in ((2.0, '-'), (3.0, '--')):
        ax.plot(RM, w2_l0(x, G), color=BLUE, ls=ls, label=r'$\ell=0$, $\Gamma=%g$' % G)
        ax.plot(RM, w2_l1(x, G), color=ORANGE, ls=ls, label=r'$\ell=1$, $\Gamma=%g$' % G)
    Rl2, g2 = catalogue_unstable(2)
    ax.plot(Rl2, -(g2*Rl2**1.5)**2, 'o', color=GRAY, ms=4, label=r'$\ell=2$ growing mode, $\Gamma=2$')
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhline(0.5, color=BLUE, lw=0.6, ls=':')
    ax.axhline(1.0, color=ORANGE, lw=0.6, ls=':')
    ax.text(11, 0.56, r'Newtonian $\ell=0$: $\Gamma-3/2$', color=BLUE, fontsize=8.5)
    ax.text(11, 1.06, r'Newtonian $\ell=1$: $(3\Gamma-4)/2$', color=ORANGE, fontsize=8.5)
    for Rm, c in ((2/(1 - (1/(np.sqrt(24) - 3))**2), ORANGE), (3.816497, BLUE)):
        ax.plot([Rm], [0], 'v', color=c, ms=6)
    ax.set_xscale('log')
    ax.set_xlim(2.05, 40)
    ax.set_ylim(-0.7, 3.3)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([2.2, 3, 4, 6, 10, 20, 40])
    ax.set_xticklabels(['2.2', '3', '4', '6', '10', '20', '40'])
    ax.set_xlabel(r'shell radius $R/M$')
    ax.set_ylabel(r'$\omega^2 R^3/M$  (exterior time)')
    ax.set_title(r'Radial and dipole modes: $\omega^2>0$ oscillates, $\omega^2<0$ grows')
    ax.legend(loc='upper left', ncol=2, fontsize=8.3)
    save(fig, 'l01_f1_omega2')


# ------------------------------------------------------------------ F2: stability diagram
def fig_stability():
    RM = np.linspace(25/12, 30, 3000)
    x = xof(RM)
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    gc = Gcaus(x)
    ax.fill_between(RM, np.maximum(G1(x), 0), np.maximum(gc, G1(x)), where=gc > G1(x), color=AQUA, alpha=0.18,
                    lw=0, label=r'$\ell=0,1$ stable and causal ($\ell\geq2$ still unstable)')
    ax.plot(RM, G1(x), color=BLUE, label=r'$\Gamma_1$: $\ell=0$ marginal (radial)')
    ax.plot(RM, Gdip(x), color=ORANGE, label=r'$\Gamma_{\rm dip}$: $\ell=1$ marginal')
    ax.plot(RM, gc, color=INK, ls='--', lw=1.3, label=r'causality $v_s^2=1$')
    ax.axhline(2, color=GRAY, lw=1.0, ls=':')
    ax.text(8, 2.1, r'$\Gamma=2$ (value used in this project)', color=GRAY, fontsize=8.5)
    ax.axvline(3, color=GRAY, lw=0.6)
    ax.text(2.95, 9.3, 'photon\nsphere', fontsize=8, color=GRAY, ha='right')
    ax.axvline(25/12, color=GRAY, lw=0.6)
    xs = NUM['x_causal_marginal']
    ax.plot([2/(1 - xs**2)], [Gcaus(xs)], 'o', color=INK, ms=4)
    ax.annotate(r'$R=%.2fM$' % (2/(1 - xs**2)), (2/(1 - xs**2), Gcaus(xs)), (2.9, 5.7), fontsize=8.5,
                arrowprops=dict(arrowstyle='->', lw=0.7))
    ax.set_xscale('log')
    ax.set_xlim(25/12, 30)
    ax.set_ylim(1, 10)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([2.2, 3, 4, 6, 10, 20, 30])
    ax.set_xticklabels(['2.2', '3', '4', '6', '10', '20', '30'])
    ax.set_xlabel(r'shell radius $R/M$')
    ax.set_ylabel(r'adiabatic index $\Gamma$')
    ax.set_title(r'Which static shells survive $\ell=0$ and $\ell=1$ perturbations')
    ax.legend(loc='upper right', fontsize=8.2)
    save(fig, 'l01_f2_stability')


# ------------------------------------------------------------------ F3: growth rates at Gamma = 2
def fig_rates():
    RM = np.linspace(25/12, 8, 2000)
    x = xof(RM)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for w2f, c, lab in ((w2_l0, BLUE, r'$\ell=0$ (radial)'), (w2_l1, ORANGE, r'$\ell=1$ (dipole)')):
        w2 = w2f(x, 2.0)
        g = np.where(w2 < 0, np.sqrt(np.abs(w2))*RM**-1.5, np.nan)
        ax.plot(RM, g, color=c, label=lab)
    for l, c in ((2, INK), (3, VIOLET)):
        Rl, gl = catalogue_unstable(l)
        ax.plot(Rl, gl, 'o-', color=c, ms=4, lw=1, label=r'$\ell=%d$ growing matter mode' % l)
    ax.set_yscale('log')
    ax.set_xlim(2.1, 8)
    ax.set_ylim(3e-3, 0.5)
    ax.set_xlabel(r'shell radius $R/M$')
    ax.set_ylabel(r'growth rate ${\rm Im}\,\omega\,M$')
    ax.set_title(r'Instability rates of the static shell, $\Gamma=2$')
    ax.legend(loc='upper right', fontsize=8.5)
    save(fig, 'l01_f3_rates')


# ------------------------------------------------------------------ F4, F5: nonlinear radial motion
def fig_nonlinear():
    d = json.load(open(os.path.join(RES, 'l0_nonlinear.json')))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for key, c, lab in (('R3_out', BLUE, r'$R_0=3M$ (unstable)'), ('R6_out', ORANGE, r'$R_0=6M$ (stable)')):
        Rg, Vg = np.array(d[key]['Vgrid_R']), np.array(d[key]['Vgrid_V'])
        ax.plot(Rg, 1e3*Vg, color=c, label=lab)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xlim(2, 12)
    ax.set_ylim(-6, 19)
    ax.annotate('barrier: a large enough inward\nkick collapses even this shell', (2.97, 16.6), (4.2, 14.5),
                fontsize=8.3, arrowprops=dict(arrowstyle='->', lw=0.7))
    ax.set_xlabel(r'$R/M$')
    ax.set_ylabel(r'$10^3\,V(R)$')
    ax.set_title(r'Effective potential, $\dot R^2+V(R)=0$ (polytrope $\Gamma=2$)')
    ax.text(2.1, -5.5, r'$\dot R^2>0$ where $V<0$', fontsize=8.5)
    ax.legend(loc='lower right')
    save(fig, 'l01_f4_potential')

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for key, c, lab in (('R3_in', BLUE, r'$R_0=3M$, pushed in by $10^{-3}$'),
                        ('R3_out', AQUA, r'$R_0=3M$, pushed out by $10^{-3}$'),
                        ('R6_in', ORANGE, r'$R_0=6M$, pushed in by $10^{-2}$')):
        tau, R = np.array(d[key]['tau']), np.array(d[key]['R'])
        ax.plot(tau, R, color=c, label=lab)
    ax.axhline(2, color=INK, lw=0.8, ls='--')
    ax.text(305, 2.12, r'horizon $r=2M$', fontsize=8.5)
    ax.plot([d['R3_in']['tau_end']], [2], 'x', color=BLUE, ms=8, mew=2)
    ax.set_xlim(0, 400)
    ax.set_ylim(1.8, 10)
    ax.set_xlabel(r'shell proper time $\tau/M$')
    ax.set_ylabel(r'$R/M$')
    ax.set_title(r'Nonlinear radial evolution of perturbed static shells, $\Gamma=2$')
    ax.legend(loc='upper right', fontsize=8.5)
    save(fig, 'l01_f5_trajectories')


# ------------------------------------------------------------------ F6: dipole mode pattern
def fig_dipole_pattern():
    A, cols, names = pickle.load(open(os.path.join(RES, 'israel_l1_matrix.pkl'), 'rb'))
    keep = [names.index(n) for n in names if n.endswith('@0') and 'phph' not in n]
    Asq = A.extract(keep, list(range(A.cols)))
    xs, Gs, Om = sp.symbols('x Gamma Omega', positive=True)
    x_, G_, Om_ = [s for s in A.free_symbols if s.name == 'x'][0], [s for s in A.free_symbols if s.name == 'Gamma'][0], \
        [s for s in A.free_symbols if s.name == 'Omega'][0]
    RM, Gv = 6.0, 2.0
    xv = np.sqrt(1 - 2/RM)
    w2 = w2_l1(xv, Gv)                               # omega^2 R^3/M (R = 1)
    Omv = np.sqrt(w2*(1 - xv**2)/2)/xv
    An = np.array(Asq.subs({x_: xv, G_: Gv, Om_: Omv}).evalf(), dtype=complex)
    _, sv, vh = np.linalg.svd(An)
    v = vh[-1].conj()
    v = v/v[cols.index([c for c in cols if c.name == 'X_p'][0])]
    X_m, a_p, X_p, b_p, s0, w0 = v
    sig0 = (1 - xv)/(4*np.pi)
    NUM['dipole_R6_G2'] = {'smallest_sv': float(sv[-1]), 'X_m/X_p': complex(X_m).real,
                           's/(sigma0 X_p)': complex(s0/sig0).real}
    print('dipole eigenvector at R=6M, Gamma=2 (X_p = 1):', NUM['dipole_R6_G2'])
    # Newtonian check of the centre of mass: X_p + s R/(3 sigma0) = 0
    for RMn in (1e4, 1e6):
        xn = np.sqrt(1 - 2/RMn)
        Omn = np.sqrt(w2_l1(xn, Gv)*(1 - xn**2)/2)/xn
        An = np.array(Asq.subs({x_: xn, G_: Gv, Om_: Omn}).evalf(30), dtype=complex)
        vn = np.linalg.svd(An)[2][-1].conj()
        vn = vn/vn[2]
        NUM['newton_com_R%.0e' % RMn] = complex(1 + vn[4]/(3*(1 - xn)/(4*np.pi))).real
        print('  R=%.0e M: X_p + s R/(3 sigma0) (X_p = 1) =' % RMn, NUM['newton_com_R%.0e' % RMn])
    # draw: exterior (centre-of-mass) frame, shell centre displaced by X_p, density dipole s
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 3.1))
    th = np.linspace(0, 2*np.pi, 400)
    amp = 0.18
    for ax, title, Xc, dens in ((axs[0], r'zero mode: rigid translation', amp, 0*th),
                                (axs[1], r'dipole oscillation (maximum)', amp,
                                 NUM['dipole_R6_G2']['s/(sigma0 X_p)']*amp*np.cos(th))):
        ax.plot(np.cos(th), np.sin(th), color=GRAY, lw=0.8, ls=':')
        zc, yc = Xc + np.cos(th), np.sin(th)
        pts = np.array([yc, zc]).T.reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap='coolwarm', norm=plt.Normalize(-0.6, 0.6), lw=5)
        lc.set_array(dens[:-1] if title.startswith('dipole') else 0*th[:-1])
        ax.add_collection(lc)
        cm = 0 if title.startswith('dipole') else Xc
        ax.plot([0], [cm], '+', color=INK, ms=10, mew=1.6)
        ax.plot([0], [Xc], '.', color=BLUE, ms=6)
        ax.set_aspect('equal')
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.35, 1.55)
        ax.axis('off')
        ax.set_title(title, fontsize=9.5)
    axs[1].text(0.08, -0.12, 'centre of mass\n(fixed)', fontsize=7.5)
    axs[1].text(-1.4, 1.35, 'lighter', color='#3b4cc0', fontsize=8)
    axs[1].text(-1.4, -1.3, 'denser', color='#b40426', fontsize=8)
    fig.text(0.5, 0.02, r'$R=6M$, $\Gamma=2$, exterior (centre-of-mass) frame, amplitude exaggerated',
             ha='center', fontsize=8)
    axs[0].text(0.08, Xc - 0.05, 'moves with\nthe shell', fontsize=7.5)
    save(fig, 'l01_f6_dipole_pattern')


if __name__ == '__main__':
    # causality meets radial stability: Gamma_1(x) = (3x+1)/(1-x)
    xx = sp.symbols('xx', positive=True)
    sol = [s for s in sp.solve(sp.Eq((1 + 2*xx + 3*xx**2)*(1 - xx), 4*xx**2*(3*xx + 1)), xx) if s.is_real and 0 < s < 1]
    NUM['x_causal_marginal'] = float(sol[0])
    NUM['R_causal_marginal'] = float(2/(1 - sol[0]**2))
    xd = sp.symbols('xd', positive=True)
    sold = [s for s in sp.solve(sp.Eq((3*xd + 1)**2*(1 - xd), 12*xd**2*(3*xd + 1)), xd) if s.is_real and 0 < s < 1]
    NUM['R_causal_marginal_dipole'] = float(2/(1 - sold[0]**2))
    print('radial stability needs acausal EOS below R =', NUM['R_causal_marginal'],
          '; dipole below R =', NUM['R_causal_marginal_dipole'])
    fig_omega2()
    fig_stability()
    fig_rates()
    fig_nonlinear()
    fig_dipole_pattern()
    json.dump(NUM, open(os.path.join(RES, 'numbers_figures.json'), 'w'), indent=1)
