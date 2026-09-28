"""Figures for the polarization / parity study (reads ../data/*.json, writes ../figures/*.pdf|png)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
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


def load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


def cplx(a):
    return np.array([complex(*x) for x in a])


def chandra_phase(w, l):
    lam = (l - 1) * l * (l + 1) * (l + 2)
    return 2 * np.arctan(12 * w / lam)


# ------------------------------------------------------------------ P1: angular patterns, l = 2, m = +-2
def fig_patterns():
    th = np.linspace(0, np.pi, 400)
    c = np.cos(th)
    Ap, Ax = 1 + c**2, 2 * np.abs(c)                 # even-parity amplitudes; odd: swapped
    fe = Ap**2 / (Ap**2 + Ax**2)
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.plot(np.degrees(th), fe, color=BLUE, label=r'even ($\Psi_{\rm even}$)')
    ax.plot(np.degrees(th), 1 - fe, color=ORANGE, label=r'odd ($\Psi_{\rm odd}$)')
    ax.axhline(0.5, color=GRAY, lw=0.7, ls=':')
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_ylim(0, 1.18)
    ax.set_xlabel(r'polar angle $\theta$ (deg)')
    ax.set_ylabel(r'fraction of the power in $h_+$')
    ax.set_title(r'$\ell=2$, $m=\pm2$: which polarization each parity produces')
    ax.text(90, 1.04, r'equator: even is pure $+$, odd is pure $\times$', ha='center', fontsize=8.8)
    ax.text(3, 0.55, 'poles: both circular', fontsize=8.8)
    ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(0.02, 0.3))
    save(fig, 'r4_f01_patterns')


# ------------------------------------------------------------------ P2: black hole = pure retarder
def fig_bh_retarder():
    d = load('bh_S.json')
    ws = np.array(d['w'])
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    wf = np.geomspace(0.02, 1.6, 300)
    errs = []
    for k, (l, col) in enumerate(((2, BLUE), (3, AQUA), (4, VIOLET))):
        So, Se = cplx(d['%d odd' % l]), cplx(d['%d even' % l])
        ph = np.angle(Se / So)
        m = np.abs(So) > 1e-3
        ax.plot(wf, np.degrees(chandra_phase(wf, l)), color=col, label=r'$\ell=%d$: $2\arctan[12M\omega/(\ell-1)\ell(\ell+1)(\ell+2)]$' % l)
        ax.plot(ws[m][::3], np.degrees(ph[m][::3]), 'o', color=col, ms=3.5, mfc='none')
        errs.append(float(np.max(np.abs(np.abs(Se[m]) / np.abs(So[m]) - 1))))
    NUM['bh_modulus_ratio_err'] = errs
    ax.axvline(0.3737, color=GRAY, lw=0.8, ls=':')
    ax.text(0.39, 150, r'${\rm Re}\,\omega_{\rm QNM}$', color=GRAY, fontsize=8.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\omega M$')
    ax.set_ylabel(r'retardance $\arg(S_{\rm even}/S_{\rm odd})$ (deg)')
    ax.set_title(r'Black hole: $|S_{\rm even}|=|S_{\rm odd}|$, but the phases differ')
    ax.legend(fontsize=7.8, loc='upper left')
    ax.set_ylim(0, 90)
    save(fig, 'r4_f02_bh_retardance')


# ------------------------------------------------------------------ P3/P4: shells: retardance and conversion
def shell_phase(key):
    d = load('shell_S_l2.json')
    so, se = d[key + ' odd'], d[key + ' even']
    wo, So = np.array(so['w']), cplx(so['S'])
    we, Se = np.array(se['w']), cplx(se['S'])
    mo, me = (wo >= 0.004) & (wo <= 1.6), (we >= 0.004) & (we <= 1.6)
    wo, So, we, Se = wo[mo], So[mo], we[me], Se[me]
    wg = np.union1d(wo, we)
    # interpolate the phases on the union grid (unwrapped)
    po = np.interp(wg, wo, np.unwrap(np.angle(So)))
    pe = np.interp(wg, we, np.unwrap(np.angle(Se)))
    return wg, np.angle(np.exp(1j * (pe - po))), pe, po


def fig_shell_retardance():
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for key, col, lab in (('6', AQUA, r'shell $R=6M$'), ('3', BLUE, r'shell $R=3M$'), ('2.2', ORANGE, r'shell $R=2.2M$'),
                          ('bh', GRAY, 'black hole')):
        wg, dph, _, _ = shell_phase(key)
        y = np.degrees(dph)
        y[np.abs(np.diff(y, prepend=y[0])) > 200] = np.nan      # do not draw the +-180 wrap
        ax.plot(wg, y, color=col, lw=2.4 if key == 'bh' else 1.3, label=lab)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlabel(r'$\omega M$')
    ax.set_ylabel(r'$\arg(S_{\rm even}/S_{\rm odd})$ (deg)')
    ax.set_title(r'Retardance of the reflected wave, $\ell=2$ (all energy is reflected: $|S|=1$)')
    ax.legend(fontsize=8.3, loc='lower left', ncol=2)
    save(fig, 'r4_f03_shell_retardance')

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for key, col, lab in (('6', AQUA, r'shell $R=6M$'), ('3', BLUE, r'shell $R=3M$'), ('2.2', ORANGE, r'shell $R=2.2M$')):
        wg, dph, _, _ = shell_phase(key)
        ax.semilogy(wg, np.sin(dph / 2)**2 + 1e-12, color=col, lw=1.3, label=lab)
    d = load('shell_S_l2.json')
    wb = np.array(d['bh odd']['w'])
    Sb_o, Sb_e = cplx(d['bh odd']['S']), cplx(d['bh even']['S'])
    ax.semilogy(wb, np.sin(chandra_phase(wb, 2) / 2)**2, color=GRAY, lw=2.4, label='black hole: fraction of the reflected wave')
    ax.semilogy(wb, np.abs(Sb_o)**2 * np.sin(chandra_phase(wb, 2) / 2)**2, ':', color=INK, lw=1.3,
                label='black hole: fraction of the incident wave')
    ax.set_xlim(0, 1.2)
    ax.set_ylim(1e-5, 1.5)
    ax.set_xlabel(r'$\omega M$')
    ax.set_ylabel(r'converted fraction $\sin^2(\Phi/2)$')
    ax.set_title(r'Polarization conversion on reflection ($\ell=2$, incident at $45^\circ$ to the E/B axes)')
    ax.legend(fontsize=7.6, loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.92)
    save(fig, 'r4_f04_conversion')
    # numbers for the text
    for key in ('6', '3', '2.2'):
        wg, dph, pe, po = shell_phase(key)
        conv = np.sin(dph / 2)**2
        i = np.argmax(conv)
        NUM['conv_max_%s' % key] = [float(wg[i]), float(conv[i])]
        mm = (wg > 0.1) & (wg < 1.0)
        NUM['conv_mean_%s_0.1_1' % key] = float(np.trapezoid(conv[mm], wg[mm]) / (wg[mm][-1] - wg[mm][0]))
        mm = (wg > 0.3) & (wg < 1.0)
        NUM['conv_max_%s_0.3_1' % key] = float(np.max(conv[mm]))
        NUM['phase_even_minus_odd_%s_at_1.0' % key] = float(np.degrees(np.interp(1.0, wg, dph)))
        NUM['phase_at_0.05_%s' % key] = float(np.degrees(np.interp(0.05, wg, dph)))


# ------------------------------------------------------------------ P5: helicity-flip cross section
def fig_sigma():
    d = load('sigma_flip.json')
    ws = np.array(d['w'])
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.loglog(ws, d['bh']['sigma'], 'o-', color=GRAY, lw=2.2, ms=3.5, label='black hole (numerical)')
    ax.loglog(ws, d['bh_analytic'], ':', color=INK, lw=1.2, label=r'black hole: $\frac{\pi}{\omega^2}\sum(2\ell+1)\mathcal{R}_\ell\sin^2\arctan\frac{12M\omega}{\lambda_\ell}$')
    for key, col, lab in (('6.0', AQUA, r'shell $R=6M$'), ('3.0', BLUE, r'shell $R=3M$'), ('2.2', ORANGE, r'shell $R=2.2M$')):
        ax.loglog(ws, d[key]['sigma'], 'o-', color=col, lw=1.3, ms=3, label=lab)
    ax.axhline(4 * np.pi / 3, color='#c43b3a', lw=0.9, ls='--')
    ax.text(0.021, 4.6, r'$4\pi M^2/3$ (long-wavelength limit)', color='#c43b3a', fontsize=8.5)
    ax.set_xlabel(r'$\omega M$')
    ax.set_ylabel(r'$\sigma_{\rm flip}/M^2$')
    ax.set_title('Helicity-reversing cross section of a circularly polarized plane wave')
    ax.legend(fontsize=7.8, loc='lower left')
    save(fig, 'r4_f05_sigma_flip')
    for k in ('bh', '6.0', '3.0', '2.2'):
        NUM['sigma_flip_%s' % k] = dict(w0=float(ws[0]), s0=float(d[k]['sigma'][0]), smax=float(np.max(d[k]['sigma'])),
                                        wmax=float(ws[np.argmax(d[k]['sigma'])]))


# ------------------------------------------------------------------ P6: time domain
def fig_time():
    d = load('timedomain.json')
    u = np.array(d['u'])
    for key, title, name, xlim in (('bh', 'Black hole', 'r4_f06_time_bh', (-50, 120)),
                                   ('3', r'Shell $R=3M$', 'r4_f07_time_R3', (-50, 200)),
                                   ('2.2', r'Shell $R=2.2M$', 'r4_f08_time_R22', (-50, 260))):
        Oo = np.array(d[key + ' odd']['causal'])
        Oe = np.array(d[key + ' even']['real_axis'])
        fig, ax = plt.subplots(figsize=(6.2, 3.3))
        ax.plot(u, Oo, color=BLUE, lw=1.5, label=r'odd channel')
        ax.plot(u, Oe, color=ORANGE, lw=1.2, label=r'even channel' + ('' if key == 'bh' else ' (stable part)'))
        if key != 'bh':
            g = np.array(d[key + ' even']['growth'])
            ax.plot(u, g, ':', color=ORANGE, lw=1.6, label='even: growing matter mode')
            NUM['u_equal_' + key] = d[key + ' even']['u_equal']
        ax.set_xlim(*xlim)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel(r'retarded time $u=t-r_*$ ($/M$)')
        ax.set_ylabel(r'reflected $\Psi(u)$ (incident peak $=1$)')
        ax.set_title(r'%s: the same incident pulse in the two parities ($\ell=2$)' % title)
        ax.legend(fontsize=8.2, loc='lower right')
        save(fig, name)
    # long-time log view: matter ringing and instability, only in the even channel
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for key, col, lw in (('6 odd', AQUA, 1.0), ('6 even', AQUA, 1.0), ('3 odd', BLUE, 1.0), ('3 even', BLUE, 1.0)):
        R = key.split()[0]
        if 'odd' in key:
            ax.semilogy(u, np.abs(np.array(d[key]['causal'])) + 1e-14, color=col, lw=0.9, ls=':', label=r'$R=%sM$ odd' % R)
        else:
            ax.semilogy(u, np.abs(np.array(d[key]['real_axis'])) + 1e-14, color=col, lw=0.9, alpha=0.7,
                        label=r'$R=%sM$ even, stable part' % R)
            ax.semilogy(u, np.abs(np.array(d[key]['causal'])) + 1e-14, color=col if R == '6' else ORANGE, lw=1.6,
                        label=r'$R=%sM$ even, full (causal)' % R)
    ax.set_xlim(-50, 700)
    ax.set_ylim(1e-9, 1e3)
    ax.set_xlabel(r'$u/M$')
    ax.set_ylabel(r'$|\Psi(u)|$ reflected')
    ax.set_title('Late times: the even channel rings at the matter frequency, then blows up')
    ax.legend(fontsize=7.6, loc='upper right', ncol=2)
    save(fig, 'r4_f09_time_log')
    # decomposition into preserved and converted polarization (incident at 45 deg)
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for key, colp, colc, lab in (('bh', GRAY, INK, 'black hole'), ('3', BLUE, ORANGE, r'$R=3M$')):
        Oo = np.array(d[key + ' odd']['causal'])
        Oe = np.array(d[key + ' even']['real_axis'])
        ax.plot(u, 0.5 * (Oe + Oo), color=colp, lw=1.0, alpha=0.55, label=lab + ': preserved polarization')
        ax.plot(u, 0.5 * (Oe - Oo), color=colc, lw=1.8, label=lab + ': rotated polarization')
        m = (u > -60) & (u < 300)
        NUM['conv_energy_' + key] = float(np.trapezoid(np.gradient(0.5 * (Oe - Oo), u)[m]**2, u[m]) /
                                          np.trapezoid(np.gradient(0.5 * (Oe + Oo), u)[m]**2 + np.gradient(0.5 * (Oe - Oo), u)[m]**2, u[m]))
    ax.set_xlim(-50, 150)
    ax.set_xlabel(r'$u/M$')
    ax.set_ylabel('reflected amplitude')
    ax.set_title(r'Incident linear polarization at $45^\circ$ to the E/B axes: what comes back rotated')
    ax.legend(fontsize=8, loc='lower right')
    save(fig, 'r4_f10_converted_time')
    # leapfrog check
    chk = d['leapfrog']
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    for key, col in (('3.0', BLUE), ('2.2', ORANGE)):
        c = chk[key]
        ax.plot(c['u'], c['leapfrog'], color=col, lw=2.4, alpha=0.4, label=r'leapfrog evolution, $R=%sM$' % key.rstrip('0').rstrip('.'))
        ax.plot(c['u'][::8], c['synth'][::8], '.', color=col, ms=2.5, label='frequency-domain synthesis')
        NUM['leapfrog_%s' % key] = [c['max_diff'], c['max_sig']]
    ax.set_xlim(-50, 250)
    ax.set_xlabel(r'$u/M$')
    ax.set_ylabel(r'$\Psi_{\rm odd}$ at the detector')
    ax.set_title('Check: time-domain evolution vs frequency-domain synthesis (odd)')
    ax.legend(fontsize=8.2)
    save(fig, 'r4_f11_leapfrog_check')
    for key in ('2.2 even', '3 even', '6 even'):
        NUM['growth_' + key] = dict(pole=d[key]['pole'], residue=d[key]['residue'], amp=d[key]['amp'], u_equal=d[key]['u_equal'])


if __name__ == '__main__':
    fig_patterns()
    fig_bh_retarder()
    if os.path.exists(os.path.join(DATA, 'shell_S_l2.json')):
        fig_shell_retardance()
    if os.path.exists(os.path.join(DATA, 'sigma_flip.json')):
        fig_sigma()
    if os.path.exists(os.path.join(DATA, 'timedomain.json')):
        fig_time()
    with open(os.path.join(HERE, 'numbers_r4.json'), 'w') as fh:
        json.dump(NUM, fh, indent=1)
    print(json.dumps(NUM, indent=1))


# ------------------------------------------------------------------ P12: absorbing (open) interior: diattenuation
def fig_open():
    path = os.path.join(HERE, '..', '..', 'First round', 'results', 'scattering.json')
    with open(path) as fh:
        sc = json.load(fh)
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for R, col in ((2.2, ORANGE), (3.0, BLUE), (6.0, AQUA)):
        co = [c for c in sc['curves'] if c['l'] == 2 and abs(c['R'] - R) < 1e-9 and c['parity'] == 'odd'][0]
        ce = [c for c in sc['curves'] if c['l'] == 2 and abs(c['R'] - R) < 1e-9 and c['parity'] == 'even'
              and c['model'] == 'C' and abs(c['Gamma'] - 2) < 1e-9][0]
        w = np.array(co['w'])
        ax.semilogy(w, np.array(ce['R1']) / np.array(co['R1']), color=col, lw=1.4, label=r'shell $R=%gM$' % R)
        NUM['open_ratio_%g' % R] = [float(np.min(np.array(ce['R1']) / np.array(co['R1']))), float(np.max(np.array(ce['R1']) / np.array(co['R1'])))]
    ax.axhline(1, color=GRAY, lw=2.2, label='black hole (isospectral)')
    ax.set_xlim(0, 2)
    ax.set_ylim(0.5, 400)
    ax.set_xlabel(r'$\omega M$')
    ax.set_ylabel(r'$\mathcal{R}_{\rm even}/\mathcal{R}_{\rm odd}$')
    ax.set_title(r'Absorbing interior: the two parities are reflected differently ($\ell=2$)')
    ax.legend(fontsize=8.3, loc='upper right')
    save(fig, 'r4_f12_open_ratio')


if __name__ == '__main__':
    fig_open()
    with open(os.path.join(HERE, 'numbers_r4.json')) as fh:
        old = json.load(fh)
    old.update(NUM)
    with open(os.path.join(HERE, 'numbers_r4.json'), 'w') as fh:
        json.dump(old, fh, indent=1)
