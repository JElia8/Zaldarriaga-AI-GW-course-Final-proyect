"""
Figures for the thin-shell GW problem (PDF + PNG in ../figures).
Reads the JSON results produced by main.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from shellgw.plots import save, C
from shellgw import studies as S
from shellgw.potentials import V as Vpot, rstar

LAB = {'odd': 'odd (RW/CPM), models A = C', 'even-C': r'even, model C (static fluid, $\Gamma=2$)',
       'even-A': 'even, model A (frozen domain wall)'}
COL = {'odd': C[0], 'even-C': C[1], 'even-A': C[2], 'BH': '0.45'}
MRK = {'odd': 'o', 'even-C': 's', 'even-A': '^'}


def fig_potentials():
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.2), sharey='row')
    for i, par in enumerate(('odd', 'even')):
        for j, R in enumerate((2.5, 6.0)):
            ax = axs[i, j]
            sf = np.sqrt(1 - 2 / R)
            xR = rstar(R)
            for k, l in enumerate((2, 3, 4)):
                r_out = np.geomspace(R, 200, 800)
                ax.plot(rstar(r_out), Vpot(r_out, l, par), color=C[k], label=r'$\ell=%d$' % l)
                r_in = np.linspace(0.12 * R, R, 400)
                x_in = xR - (R - r_in) / sf
                ax.plot(x_in, sf**2 * l * (l + 1) / r_in**2, color=C[k], ls='-')
                rb = np.geomspace(2.0005, 200, 1500)
                ax.plot(rstar(rb), Vpot(rb, l, par), color=C[k], ls=':', lw=0.9)
            ax.axvline(xR, color='k', ls='--', lw=0.8)
            ax.text(xR, 0.03, r'  shell $r=R$', fontsize=8, rotation=90, va='bottom')
            ax.set_xlim(xR - 0.9 * R / sf, 25)
            ax.set_ylim(0, 1.2 if par == 'odd' else 1.2)
            ax.set_title(r'%s parity, $R=%.1fM$' % (par, R))
            if i == 1:
                ax.set_xlabel(r'$x/M$  ($x=r_*$ outside; $x-x_R=(r-R)/\sqrt{f_R}$ inside)')
            if j == 0:
                ax.set_ylabel(r'$V\,M^2$')
            if i == 0 and j == 0:
                ax.legend(loc='upper right', title='solid: shell spacetime\ndotted: Schwarzschild BH', fontsize=8,
                          title_fontsize=7.5)
    fig.suptitle(r'Effective potentials: exterior $V_{\rm RW}$, $V_{\rm Z}$; flat interior $f_R\,\ell(\ell+1)/r^2$;'
                 r' shell = repulsive $\delta$-coupling (odd: $\Delta=\sqrt{f_R}(1-\sqrt{f_R})/R$)', fontsize=9.5)
    fig.tight_layout()
    save(fig, 'fig01_potentials')


def _curves(data, l, R, model=None, par=None, Gamma=S.GAMMA_FID):
    out = []
    for c in data['curves']:
        if c['l'] == l and abs(c['R'] - R) < 1e-9 and (par is None or c['parity'] == par) \
                and (model is None or c['model'] == model) and abs(c['Gamma'] - Gamma) < 1e-9:
            out.append(c)
    return out


def _bh(data, l, par='odd'):
    for b in data['bh']:
        if b['l'] == l and b['parity'] == par:
            return b


def fig_RT(data, l, Rs, name):
    fig, axs = plt.subplots(2, len(Rs), figsize=(3.1 * len(Rs), 5.6), sharex=True)
    for j, R in enumerate(Rs):
        axT, axB = axs[0, j], axs[1, j]
        for par, model in (('odd', 'C'), ('even', 'C'), ('even', 'A')):
            cs = _curves(data, l, R, model, par)
            if not cs:
                continue
            c = cs[0]
            key = S.cfg_label(par, model)
            w = np.array(c['w'])
            axT.plot(w, c['R1'], color=COL[key], label=LAB[key] if j == 0 else None)
            axT.plot(w, c['T1'], color=COL[key], ls='--')
            axB.semilogy(w, c['R1'], color=COL[key])
        b = _bh(data, l)
        axT.plot(b['w'], b['R'], color=COL['BH'], lw=1, ls=':', label='Schwarzschild BH (Vishveshwara 1970)' if j == 0 else None)
        axB.semilogy(b['w'], b['R'], color=COL['BH'], lw=1, ls=':')
        axT.set_title(r'$\ell=%d$, $R=%gM$' % (l, R))
        axB.set_xlabel(r'$\omega M$')
        axB.set_ylim(1e-7, 2)
        axT.set_ylim(-0.02, 1.02)
    axs[0, 0].set_ylabel(r'$\mathcal{R}$ (solid), $\mathcal{T}$ (dashed)')
    axs[1, 0].set_ylabel(r'$\mathcal{R}(\omega)$')
    fig.legend(loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, name)


def fig_gamma(data):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.3))
    for k, G in enumerate((1.6, 2.0, 3.0)):
        cs = _curves(data, 2, 3.0, 'C', 'even', G)
        if cs:
            c = cs[0]
            ax[0].plot(c['w'], c['R1'], color=C[k], label=r'$\Gamma=%.1f$ ($v_s^2=%.3f$)' % (G, c['vs2']))
            ax[1].semilogy(c['w'], c['R1'], color=C[k])
    c = _curves(data, 2, 3.0, 'C', 'odd')[0]
    ax[0].plot(c['w'], c['R1'], color='k', ls=':', label='odd parity')
    ax[1].semilogy(c['w'], c['R1'], color='k', ls=':')
    for a in ax:
        a.set_xlabel(r'$\omega M$')
    ax[0].set_ylabel(r'$\mathcal{R}$')
    ax[0].legend(fontsize=8)
    ax[0].set_title(r'even parity, model C, $\ell=2$, $R=3M$: EOS dependence')
    ax[1].set_ylim(1e-6, 2)
    fig.tight_layout()
    save(fig, 'fig04_eos_dependence')


def fig_tm_vs_shoot(data):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True)
    for j, (par, model) in enumerate((('odd', 'C'), ('even', 'C'), ('even', 'A'))):
        key = S.cfg_label(par, model)
        for k, R in enumerate((3.0, 6.0)):
            c = _curves(data, 2, R, model, par)[0]
            w = np.array(c['w'])
            axs[0, j].plot(w, c['R1'], color=C[k], lw=2.2, alpha=0.5, label=r'M1 transfer matrix, $R=%gM$' % R)
            axs[0, j].plot(w[::4], np.array(c['R2'])[::4], 'o', ms=3, color=C[k], mfc='none',
                           label='M2 shooting' if k == 0 else None)
            axs[0, j].plot(w[::4], np.array(c['Rbw'])[::4], 'x', ms=3, color=C[k],
                           label='Boyanov et al. windowed estimator' if k == 0 else None)
            axs[1, j].semilogy(w, np.abs(np.array(c['R1']) - np.array(c['R2'])) + 1e-18, color=C[k], label=r'$|\Delta\mathcal{R}|$, $R=%gM$' % R)
            axs[1, j].semilogy(w, np.abs(np.array(c['T1']) - np.array(c['T2'])) + 1e-18, color=C[k], ls='--', label=r'$|\Delta\mathcal{T}|$')
        axs[0, j].set_title(LAB[key], fontsize=9)
        axs[1, j].axhline(1e-4, color='r', lw=0.8, ls=':')
        axs[1, j].set_xlabel(r'$\omega M$')
        axs[1, j].set_ylim(1e-17, 1e-2)
    axs[0, 0].set_ylabel(r'$\mathcal{R}(\omega)$, $\ell=2$')
    axs[1, 0].set_ylabel('method difference')
    axs[0, 0].legend(fontsize=7)
    axs[1, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, 'fig05_transfer_vs_shooting')


def fig_energy(data):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for (par, model) in (('odd', 'C'), ('even', 'C'), ('even', 'A')):
        key = S.cfg_label(par, model)
        for k, R in enumerate((3.0, 6.0)):
            c = _curves(data, 2, R, model, par)[0]
            w = np.array(c['w'])
            e2 = np.abs(np.array(c['R2']) + np.array(c['T2']) - 1) + 1e-18
            e1 = np.abs(np.array(c['R1']) + np.array(c['T1']) - 1) + 1e-18
            ls = '-' if k == 0 else '--'
            ax[0].semilogy(w, e2, color=COL[key], ls=ls, label='%s, $R=%gM$' % (key, R))
            ax[1].loglog(w, e1, color=COL[key], ls=ls)
    for a in ax:
        a.axhline(1e-6, color='r', lw=0.8, ls=':')
        a.set_xlabel(r'$\omega M$')
    ax[0].set_ylabel(r'$|\mathcal{R}+\mathcal{T}-1|$ (method 2, independent)')
    ax[1].set_ylabel(r'$|\mathcal{R}+\mathcal{T}-1|$ (method 1)')
    ww = np.geomspace(0.5, 2, 10)
    ax[1].loglog(ww, 3e-2 * ww**-2, 'k:', lw=1)
    ax[1].text(0.9, 4e-2, r'$\propto\omega^{-2}$', fontsize=8)
    ax[0].legend(fontsize=7, ncol=1)
    ax[0].set_title(r'Energy conservation, $\ell=2$')
    ax[1].set_title('model A: violation is an adiabatic-approximation error')
    fig.tight_layout()
    save(fig, 'fig08_energy_conservation')


def fig_bh_limit(data):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    b = _bh(data, 2)
    for i, par in enumerate(('odd', 'even')):
        a = ax[i]
        a.plot(b['w'], b['R'], color='k', lw=2.5, alpha=0.4, label='Schwarzschild BH')
        for k, R in enumerate((3.0, 2.2, 2.05, 2.01)):
            cs = _curves(data, 2, R, 'C', par)
            if cs:
                a.plot(cs[0]['w'], cs[0]['R1'], color=C[k], label=r'shell $R=%gM$' % R)
        a.set_xlim(0, 1)
        a.set_xlabel(r'$\omega M$')
        a.set_title('%s parity, $\\ell=2$ (even: model C)' % par)
    ax[0].set_ylabel(r'$\mathcal{R}(\omega)$')
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'fig09_schwarzschild_limit')


def _cat(cat, l, par, model, R):
    for c in cat:
        if c['l'] == l and c['parity'] == par and c['model'] == model and abs(c['R'] - R) < 1e-9:
            return c


def fig_qnm_plane(cat, V):
    Rs = (2.2, 3.0, 6.0, 10.0)
    fig, axs = plt.subplots(1, len(Rs), figsize=(3.2 * len(Rs), 3.9), sharey=True)
    bh = [b for b in V['bh_qnm'] if b['l'] == 2]
    for j, R in enumerate(Rs):
        ax = axs[j]
        for par, model in (('odd', 'C'), ('even', 'C'), ('even', 'A')):
            key = S.cfg_label(par, model)
            c = _cat(cat, 2, par, model, R)
            if not c:
                continue
            ws = np.array([m['w1'] for m in c['modes']])
            if len(ws):
                ax.plot(ws[:, 0], ws[:, 1], MRK[key], color=COL[key], ms=5, mfc='none',
                        label=LAB[key] if j == 0 else None)
        ax.plot([b['leaver'][0] for b in bh], [b['leaver'][1] for b in bh], '*', color='k', ms=8,
                label=r'Schwarzschild BH ($n=0..3$)' if j == 0 else None)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlim(-0.03, 2.65)
        ax.set_ylim(-1.35, 0.3)
        ax.set_title(r'$\ell=2$, $R=%gM$' % R)
        ax.set_xlabel(r'${\rm Re}\,\omega M$')
    axs[0].set_ylabel(r'${\rm Im}\,\omega M$')
    fig.legend(loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, 'fig06_qnm_complex_plane')


def fig_qnm_vs_R(tracks, V, cat):
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True)
    bh0 = [b for b in V['bh_qnm'] if b['l'] == 2 and b['n'] == 0][0]['leaver']
    for tr in tracks:
        if not tr['pts']:
            continue
        key = S.cfg_label(tr['parity'], tr['model'])
        Rv = np.array([p['R'] for p in tr['pts']])
        wv = np.array([p['w'] for p in tr['pts']])
        matter = abs(tr['start'][0]) < 0.25 and (tr['model'] == 'C' and tr['parity'] == 'even') and \
            (abs(tr['start'][1]) < 0.02 or abs(tr['start'][0]) < 1e-6)
        col = 0 if tr['parity'] == 'odd' else 1
        order = np.argsort(Rv)
        if matter:
            sc = np.sqrt(1 / Rv**3)
            axs[0, col].plot(Rv[order], (wv[:, 0] / sc)[order] * 0 + wv[:, 0][order], color=C[3], lw=1)
            axs[1, col].plot(Rv[order], wv[:, 1][order], color=C[3], lw=1)
        else:
            axs[0, col].plot(Rv[order], wv[:, 0][order], color=COL[key], lw=1.2)
            axs[1, col].plot(Rv[order], wv[:, 1][order], color=COL[key], lw=1.2)
    for col, par in enumerate(('odd', 'even')):
        for R in (2.2, 2.5, 3.0, 4.0, 6.0, 10.0):
            c = _cat(cat, 2, par, 'C', R)
            if c:
                ws = np.array([m['w1'] for m in c['modes']])
                axs[0, col].plot([R] * len(ws), ws[:, 0], '.', color='k', ms=3)
                axs[1, col].plot([R] * len(ws), ws[:, 1], '.', color='k', ms=3)
        for a in axs[:, col]:
            a.set_xscale('log')
            a.set_xticks([2, 3, 4, 6, 10, 20, 40])
            a.set_xticklabels(['2', '3', '4', '6', '10', '20', '40'])
            a.minorticks_off()
            a.axvline(3.0, color='0.6', lw=0.7, ls='--')
        axs[0, col].axhline(bh0[0], color='k', ls=':', lw=1)
        axs[1, col].axhline(bh0[1], color='k', ls=':', lw=1)
        axs[0, col].set_title('%s parity, $\\ell=2$%s' % (par, ' (model C, $\\Gamma=2$)' if par == 'even' else ''))
        axs[1, col].set_xlabel(r'$R/M$')
        axs[0, col].set_ylim(0, 1.5)
        axs[1, col].set_ylim(-0.8, 0.1)
    axs[0, 0].set_ylabel(r'${\rm Re}\,\omega M$')
    axs[1, 0].set_ylabel(r'${\rm Im}\,\omega M$')
    axs[0, 0].text(3.05, 1.4, 'photon sphere', fontsize=7, color='0.4')
    axs[0, 1].text(0.02, 0.93, 'pink: matter modes (Pitre et al. 2026)\ndotted: BH $\\ell=2,n=0$', transform=axs[0, 1].transAxes,
                   fontsize=7, va='top')
    fig.tight_layout()
    save(fig, 'fig07_qnm_vs_compactness')


def fig_pitre(V):
    P = V['pitre2026_matter']
    rows = P['rows']
    MR = np.array([r_['M_over_R'] for r_ in rows])
    un = np.array([r_['unstable'][1] for r_ in rows])
    st = np.array([r_['stable'][0] for r_ in rows])
    sti = np.array([r_['stable'][1] for r_ in rows])
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    ax[0].plot(MR, un, 'o-', color=C[1], label='this work')
    ax[0].plot(0, P['pn_varsigma0_imag'], '*', color='k', ms=9, label='PN limit, Pitre et al. Eq. (7.8)')
    ax[0].set_ylabel(r'${\rm Im}\,\omega\,(R^3/M)^{1/2}$ (unstable)')
    ax[1].plot(MR, st, 's-', color=C[0])
    ax[1].plot(0, P['pn_varsigma0_real'], '*', color='k', ms=9)
    ax[1].set_ylabel(r'${\rm Re}\,\omega\,(R^3/M)^{1/2}$ (stable pair)')
    ax[2].plot(MR, sti * 1e3, 's-', color=C[0])
    ax[2].set_ylabel(r'$10^3\,{\rm Im}\,\omega\,(R^3/M)^{1/2}$ (stable pair)')
    for a in ax:
        a.set_xlabel(r'$M/R$')
    ax[0].legend(fontsize=7)
    fig.suptitle(r'Even-parity matter modes of the static fluid shell (model C), $\ell=2$, $\Gamma=2$ '
                 r'-- compare Pitre, Schneider & Poisson (2026) Figs. 1, 2', fontsize=9)
    fig.tight_layout()
    save(fig, 'fig10_matter_modes_pitre')


def fig_modelA(V, data):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    rows = V['junction']['modelA_constraint']
    for k, R in enumerate((3.0, 6.0)):
        r_ = [x for x in rows if x['R'] == R]
        ax[0].loglog([x['w'] for x in r_], [x['constr'] for x in r_], 'o-', color=C[k], label=r'$R=%gM$' % R)
    ww = np.geomspace(0.3, 100, 10)
    ax[0].loglog(ww, 0.1 * ww**-1, 'k:', lw=1)
    ax[0].text(3, 0.05, r'$\propto\omega^{-1}$', fontsize=8)
    ax[0].set_xlabel(r'$\omega M$')
    ax[0].set_ylabel(r'relative violation of $\tau\tau$, $\tau A$ Israel eqs.')
    ax[0].legend(fontsize=8)
    ax[0].set_title('model A (frozen domain wall): constraint residual')
    for k, R in enumerate((3.0, 6.0)):
        cA = _curves(data, 2, R, 'A', 'even')[0]
        cC = _curves(data, 2, R, 'C', 'even')[0]
        ax[1].plot(cA['w'], cA['R1'], color=C[k], label=r'A, $R=%gM$' % R)
        ax[1].plot(cC['w'], cC['R1'], color=C[k], ls='--', label=r'C, $R=%gM$' % R)
    ax[1].set_xlabel(r'$\omega M$')
    ax[1].set_ylabel(r'$\mathcal{R}$ (even, $\ell=2$)')
    ax[1].legend(fontsize=8)
    ax[1].set_title('even parity: model A vs model C')
    fig.tight_layout()
    save(fig, 'fig11_modelA_adiabatic')


def fig_wkb(V):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, order in enumerate((1, 3, 6)):
        errs = [abs(complex(*b['wkb%d' % order]) - complex(*b['leaver'])) / abs(complex(*b['leaver']))
                for b in V['bh_qnm']]
        labels = ['%d,%d' % (b['l'], b['n']) for b in V['bh_qnm']]
        ax[0].semilogy(range(len(errs)), errs, 'o-', color=C[k], label='WKB order %d' % order)
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_xticklabels(labels, rotation=90, fontsize=7)
    ax[0].set_xlabel(r'$(\ell,n)$')
    ax[0].set_ylabel(r'$|\omega_{\rm WKB}-\omega_{\rm Leaver}|/|\omega|$')
    ax[0].legend(fontsize=8)
    ax[0].set_title('Schwarzschild QNMs: WKB vs Leaver')
    tw = V['trapped_wkb']
    for k, R in enumerate(sorted(set(t['R'] for t in tw))):
        t_ = [t for t in tw if t['R'] == R and t['res'] < 1e-8]
        ax[1].semilogy([t['exact'][0] for t in t_], [-t['exact'][1] for t in t_], 'o', color=C[k], label=r'exact, $R=%gM$' % R)
        ax[1].semilogy([t['wkb'][0] for t in t_], [-t['wkb'][1] for t in t_], 'x', color=C[k])
    ax[1].set_xlabel(r'${\rm Re}\,\omega M$')
    ax[1].set_ylabel(r'$-{\rm Im}\,\omega M$')
    ax[1].set_title('trapped modes (odd, $\\ell=2$): exact (o) vs WKB (x)')
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, 'fig12_wkb')


def main():
    data = S._load('scattering.json')
    V = S._load('validation.json')
    cat = S._load('qnm_catalogue.json')
    tracks = S._load('qnm_tracks.json')
    fig_potentials()
    if data:
        fig_RT(data, 2, (2.2, 3.0, 6.0, 10.0), 'fig02_RT_l2')
        fig_RT(data, 3, (3.0, 6.0), 'fig03a_RT_l3')
        fig_RT(data, 4, (3.0, 6.0), 'fig03b_RT_l4')
        fig_gamma(data)
        fig_tm_vs_shoot(data)
        fig_energy(data)
        fig_bh_limit(data)
    if cat and V:
        fig_qnm_plane(cat, V)
    if tracks and V and cat:
        fig_qnm_vs_R(tracks, V, cat)
    if V:
        fig_pitre(V)
        fig_wkb(V)
        if data:
            fig_modelA(V, data)


if __name__ == '__main__':
    main()

