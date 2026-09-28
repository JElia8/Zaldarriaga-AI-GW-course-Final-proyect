"""
Third-round figures: single-panel, uniform style, written to ../figures (PDF + PNG).

Data sources (read only):
  ../../First round/results/*.json        scattering curves, QNM catalogue, validation data
  ../../Second round/checks/out_*.json    thick-shell test, model-A prescriptions
  fr_code/                                a copy of the first-round code (used to compute
                                          a few extra R(omega) points and nothing else)
New computation done here:
  * R(omega) up to omega M = 10 (high-frequency delta-barrier law)
  * time-domain evolution of a wave packet (odd parity, l = 2) for a black hole and for shells,
    with a fit of the late-time ringing to the frequency-domain QNM.
Run:  python -B make_figures_r3.py
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import lambertw

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
FIG = os.path.join(HERE, '..', 'figures')
FR = os.path.join(ROOT, 'First round', 'results')
SR = os.path.join(ROOT, 'Second round', 'checks')
sys.path.insert(0, os.path.join(HERE, 'fr_code'))
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, PINK, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7'
GRAY, INK = '#8a8984', '#222222'
SEQ = [VIOLET, BLUE, AQUA, YELLOW]          # for families ordered by compactness
plt.rcParams.update({'font.family': 'serif', 'font.size': 10.5, 'axes.labelsize': 11.5, 'legend.fontsize': 9,
                     'axes.titlesize': 11, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'lines.linewidth': 1.8,
                     'savefig.bbox': 'tight', 'mathtext.fontset': 'cm', 'legend.frameon': False})
OUT = {}                                     # numbers quoted in the documents


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, name + '.' + ext), dpi=230 if ext == 'png' else None)
    plt.close(fig)
    print('saved', name)


def load(path):
    with open(path) as fh:
        return json.load(fh)


scat = load(os.path.join(FR, 'scattering.json'))
cat = load(os.path.join(FR, 'qnm_catalogue.json'))
V = load(os.path.join(FR, 'validation.json'))


def curve(l, R, par, model='C', Gamma=2.0):
    for c in scat['curves']:
        if c['l'] == l and abs(c['R'] - R) < 1e-9 and c['parity'] == par and c['model'] == model \
                and abs(c['Gamma'] - Gamma) < 1e-9:
            return c


def bh(l=2, par='odd'):
    for b in scat['bh']:
        if b['l'] == l and b['parity'] == par:
            return b


def modes(l, R, par, model='C'):
    for c in cat:
        if c['l'] == l and abs(c['R'] - R) < 1e-9 and c['parity'] == par and c['model'] == model:
            return c['modes']
    return []


def rstar(r, M=1.0):
    return r + 2 * M * np.log(r / (2 * M) - 1)


def r_of_x(x, M=1.0):
    """Inverse tortoise coordinate, r = 2M[1 + W(exp(x/2M - 1))]."""
    return 2 * M * (1 + np.real(lambertw(np.exp(x / (2 * M) - 1))))


def V_RW(r, l, M=1.0):
    f = 1 - 2 * M / r
    return f * (l * (l + 1) / r**2 - 6 * M / r**3)


def V_Z(r, l, M=1.0):
    n = (l - 1) * (l + 2) / 2
    f = 1 - 2 * M / r
    return f * (2 * n**2 * (n + 1) * r**3 + 6 * n**2 * M * r**2 + 18 * n * M**2 * r + 18 * M**3) / (r**3 * (n * r + 3 * M)**2)


# ============================================================ F1 potential
def fig_potential():
    R, l = 3.0, 2
    s = np.sqrt(1 - 2 / R)
    xR = rstar(R)
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    rin = np.linspace(0.18 * R, R, 400)
    xin = xR - (R - rin) / s
    rout = np.geomspace(R, 300, 1500)
    rbh = np.geomspace(2.0005, R, 800)
    ax.fill_between(xin, 0, 1, color=BLUE, alpha=0.06, lw=0)
    ax.plot(xin, s**2 * l * (l + 1) / rin**2, color=INK, lw=1.8, label=r'interior: $f_R\,\ell(\ell+1)/r^2$ (flat space)')
    ax.plot(rstar(rout), V_RW(rout, l), color=BLUE, label=r'exterior, odd: $V_{\rm RW}$')
    ax.plot(rstar(rout), V_Z(rout, l), color=ORANGE, label=r'exterior, even: $V_{\rm Z}$')
    ax.plot(rstar(rbh), V_RW(rbh, l), ':', color=GRAY, lw=1.3, label='black hole (continuation to $r=2M$)')
    ax.annotate('', xy=(xR, 0.235), xytext=(xR, 0.0), arrowprops=dict(arrowstyle='-|>', color=INK, lw=2.2))
    ax.text(xR + 0.6, 0.20, 'shell: repulsive\n' + r'$\delta$-barrier', fontsize=9)
    ax.text(xR - 7.2, 0.215, 'flat\ninterior', fontsize=9, color=BLUE)
    ax.set_xlim(xR - 0.85 * R / s, 22)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel(r'global tortoise coordinate $x/M$')
    ax.set_ylabel(r'$V\,M^2$')
    ax.set_title(r'Effective potential seen by the wave ($\ell=2$, $R=3M$)')
    ax.legend(loc='upper right', fontsize=8.3)
    save(fig, 'r3_f01_potential')


# ============================================================ F2 reflectivity, odd, vs compactness
def fig_reflectivity():
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    b = bh()
    ax.plot(b['w'], b['R'], color='k', lw=3, alpha=0.25, label='Schwarzschild black hole')
    for k, R in enumerate((2.2, 3.0, 6.0, 10.0)):
        c = curve(2, R, 'odd')
        ax.plot(c['w'], c['R1'], color=SEQ[k], label=r'shell $R=%gM$' % R)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel(r'$\omega M$')
    ax.set_ylabel(r'reflectivity $\mathcal{R}(\omega)$')
    ax.set_title(r'Odd parity, $\ell=2$: opaque at low $\omega$, transparent at high $\omega$')
    ax.legend()
    save(fig, 'r3_f02_reflectivity_odd')


# ============================================================ F3 high-frequency law, odd vs even
def fig_highfreq():
    from shellgw.scattering import barrier_RT
    from shellgw.junction import Delta_odd
    R = 3.0
    s = np.sqrt(1 - 2 / R)
    k = (1 - s) / (4 * s)
    vs2 = 2 * k / (1 + k)                       # Gamma = 2
    c_o, c_e = curve(2, R, 'odd'), curve(2, R, 'even')
    wx = np.geomspace(2.2, 12, 14)
    Ro = np.array([barrier_RT(w, 2, 'odd', R)[0] for w in wx])
    Re = np.array([barrier_RT(w, 2, 'even', R, model='C', vs2=vs2)[0] for w in wx])
    D = Delta_odd(R)
    OUT['highfreq'] = dict(w=list(wx), R_odd=list(Ro), R_even=list(Re), delta_law=list(D**2 / (4 * wx**2 + D**2)))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.loglog(c_o['w'], c_o['R1'], color=BLUE, label='odd parity')
    ax.loglog(wx, Ro, 'o', color=BLUE, ms=3.5)
    ax.loglog(c_e['w'], c_e['R1'], color=ORANGE, label=r'even parity (fluid shell, $\Gamma=2$)')
    ax.loglog(wx, Re, 's', color=ORANGE, ms=3.5, mfc='none')
    ww = np.geomspace(0.3, 12, 100)
    ax.loglog(ww, D**2 / (4 * ww**2 + D**2), '--', color=INK, lw=1.2,
              label=r'bare $\delta$-barrier: $\Delta^2/(4\omega^2+\Delta^2)$')
    b = bh()
    ax.loglog(b['w'], b['R'], color='k', lw=3, alpha=0.2, label='black hole')
    ax.set_ylim(1e-6, 2)
    ax.set_xlim(0.02, 12)
    ax.set_xlabel(r'$\omega M$')
    ax.set_ylabel(r'$\mathcal{R}(\omega)$')
    ax.set_title(r"$R=3M$, $\ell=2$: at high frequency only the shell's $\delta$-barrier remains")
    ax.legend(loc='lower left', fontsize=8.3)
    save(fig, 'r3_f03_highfreq_law')


# ============================================================ F4 black-hole limit
def fig_bh_limit():
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    b = bh()
    ax.plot(b['w'], b['R'], color='k', lw=3.2, alpha=0.3, label='Schwarzschild black hole (Vishveshwara)')
    for k, R in enumerate((3.0, 2.2, 2.05, 2.01)):
        c = curve(2, R, 'odd')
        ax.plot(c['w'], c['R1'], color=SEQ[::-1][k], lw=1.5, label=r'shell $R=%gM$' % R)
    ax.set_xlim(0.15, 0.75)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel(r'$\omega M$')
    ax.set_ylabel(r'$\mathcal{R}(\omega)$')
    ax.set_title(r'Odd parity, $\ell=2$: shell $\to$ black hole as $R\to 2M$')
    ax.legend(fontsize=8.3)
    save(fig, 'r3_f04_bh_limit')


# ============================================================ F5 QNMs in the complex plane
def fig_qnm_plane():
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    for k, R in enumerate((2.2, 3.0, 6.0, 10.0)):
        ws = np.array([m['w1'] for m in modes(2, R, 'odd')])
        ax.plot(ws[:, 0], ws[:, 1], 'o', color=SEQ[k], ms=5, mfc='none', mew=1.4, label=r'shell $R=%gM$' % R)
    bq = np.array([b['leaver'] for b in V['bh_qnm'] if b['l'] == 2])
    ax.plot(bq[:, 0], bq[:, 1], '*', color='k', ms=10, label=r'black hole, $n=0\ldots3$')
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-1.1, 0.03)
    ax.set_xlabel(r'${\rm Re}\,\omega M$')
    ax.set_ylabel(r'${\rm Im}\,\omega M$')
    ax.set_title(r'Odd-parity QNMs, $\ell=2$: the shell does not ring like a black hole')
    ax.legend(loc='lower right', fontsize=8.3, ncol=1)
    save(fig, 'r3_f05_qnm_plane')


# ============================================================ F6 matter modes: instability
def fig_matter():
    rows = V['pitre2026_matter']['rows']
    MR = np.array([r['M_over_R'] for r in rows])
    un = np.array([r['unstable'][1] for r in rows])
    st = np.array([r['stable'][0] for r in rows])
    psp_MR = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])      # read off PSP26 Figs. 1-2
    psp_un = np.array([0.518, 0.535, 0.557, 0.582, 0.608, 0.637, 0.668])
    psp_st = np.array([1.495, 1.454, 1.398, 1.336, 1.267, 1.188, 1.096])
    # Newtonian shell (paper App. A): varsigma^4 - S varsigma^2 + P = 0, l = 2, Gamma = 2
    l, G = 2, 2.0
    S = ((l * l + l + 4) * G - (l * l + l + 6)) / 4
    P = -(l - 1) * l * (l + 1) * ((l + 2) * (2 * l - 3) * G - 4 * (l - 2)) / (16 * (2 * l + 1))
    roots = np.roots([1, -S, P])
    nst, nun = np.sqrt(roots.max()), np.sqrt(-roots.min())
    OUT['newtonian'] = dict(stable=nst, unstable=nun)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(MR, un, 'o-', color=ORANGE, ms=4.5, label=r'growing mode: ${\rm Im}\,\omega$ (this work)')
    ax.plot(MR, st, 's-', color=BLUE, ms=4.5, label=r'oscillating pair: ${\rm Re}\,\omega$ (this work)')
    ax.plot(psp_MR, psp_un, 'x', color=INK, ms=7, mew=1.4, label='Pitre, Schneider & Poisson (2026)')
    ax.plot(psp_MR, psp_st, 'x', color=INK, ms=7, mew=1.4)
    ax.plot([0, 0], [nun, nst], '*', color=YELLOW, ms=13, mec=INK, mew=0.6, label='Newtonian shell (analytic)')
    ax.set_xlabel(r'compactness $M/R$')
    ax.set_ylabel(r'$\omega\,(R^3/M)^{1/2}$')
    ax.set_xlim(-0.012, 0.31)
    ax.set_ylim(0.4, 1.65)
    ax.set_title(r'Even-parity matter modes of the static fluid shell ($\ell=2$, $\Gamma=2$)')
    ax.legend(loc='center right', fontsize=8.3)
    save(fig, 'r3_f06_matter_modes')


# ============================================================ F7 time scales
def fig_timescales():
    Rg = np.geomspace(2.1, 40, 200)
    s = np.sqrt(1 - 2 / Rg)
    wdyn = s * np.sqrt((1 + 3 * s) / (2 * Rg) / Rg)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.loglog(Rg, wdyn, color=AQUA, lw=2.2, label=r'domain-wall collapse rate $\omega_{\rm dyn}$ (model A)')
    ax.loglog(Rg, 0.6 * Rg**-1.5, color=ORANGE, lw=2.2, label=r'fluid-shell instability rate $\approx0.6(M/R^3)^{1/2}$ (model C)')
    xs, ys = [], []
    for c in cat:
        if c['parity'] == 'odd' and c['l'] == 2:
            wv = sorted([m for m in c['modes'] if m['kind'] == 'wave'], key=lambda m: -m['w1'][1])
            if wv:
                xs.append(c['R'])
                ys.append(abs(complex(*wv[0]['w1'])))
    ax.loglog(xs, ys, 'o', color=BLUE, ms=6, mfc='none', mew=1.5, label=r'$|\omega|$ of the longest-lived wave QNM')
    ax.loglog(Rg, 1 / Rg, ':', color=GRAY, label=r'$1/R$')
    ax.set_xticks([2, 3, 4, 6, 10, 20, 40])
    ax.set_xticklabels(['2', '3', '4', '6', '10', '20', '40'])
    ax.minorticks_off()
    ax.set_xlabel(r'$R/M$')
    ax.set_ylabel(r'rate $\times M$')
    ax.set_title('Is the background slow compared with the waves?')
    ax.legend(fontsize=8.2, loc='lower left')
    save(fig, 'r3_f07_timescales')


# ============================================================ F8 validation
def fig_validation():
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for par, col, lab in (('odd', BLUE, 'odd'), ('even', ORANGE, 'even (model C)')):
        c = curve(2, 3.0, par)
        w = np.array(c['w'])
        e = np.abs(np.array(c['R2']) + np.array(c['T2']) - 1) + 1e-18
        d = np.abs(np.array(c['R1']) - np.array(c['R2'])) + 1e-18
        ax.semilogy(w, e, color=col, label=r'$|\mathcal{R}+\mathcal{T}-1|$, %s' % lab)
        ax.semilogy(w, d, '--', color=col, lw=1.2, label=r'$|\mathcal{R}_{\rm M1}-\mathcal{R}_{\rm M2}|$, %s' % lab)
    ax.axhline(1e-6, color='#c43b3a', lw=0.9, ls=':')
    ax.text(1.35, 1.6e-6, 'required accuracy', color='#c43b3a', fontsize=8.5)
    ax.set_ylim(1e-17, 1e-4)
    ax.set_xlabel(r'$\omega M$')
    ax.set_ylabel('error')
    ax.set_title(r'Energy balance and method agreement ($\ell=2$, $R=3M$)')
    ax.legend(fontsize=8, loc='lower right', ncol=1)
    save(fig, 'r3_f08_validation')


# ============================================================ F9 thick-shell test
def fig_thick():
    d = load(os.path.join(SR, 'out_c03b.json'))
    rows = [r for r in d if abs(r['w'] - 0.5) < 1e-9]
    eps = np.array([r['eps'] for r in rows])
    keys = [k for k in rows[0] if k not in ('w', 'eps', 'Phi_in_minus_half_ln_f')]
    labs = [r'$\Delta=\sqrt{f_R}(1-\sqrt{f_R})/R$ (derived)', r'$\Delta=0$ (no shell)',
            r'$\Delta=(1-\sqrt{f_R})/R$', r'$2\times$ derived $\Delta$']
    cols = [BLUE, GRAY, ORANGE, YELLOW]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for i, k in enumerate(keys):
        ax.loglog(eps, [r[k] for r in rows], 'o-', color=cols[i], ms=4, lw=2.4 if i == 0 else 1.2, label=labs[i])
    ax.loglog(eps, 0.09 * eps, ':', color=INK, lw=1)
    ax.text(0.01, 0.0005, r'$\propto\varepsilon$', fontsize=10)
    ax.set_xlabel(r'half-thickness of a smooth shell $\varepsilon/M$')
    ax.set_ylabel('mismatch with thin-shell junction')
    ax.set_title(r'Odd junction from a thick-shell limit ($\omega M=0.5$, $\ell=2$, $R=3M$)')
    ax.legend(fontsize=8.3, loc='lower right')
    save(fig, 'r3_f09_thick_shell')


# ============================================================ F10 model A ambiguity
def fig_modelA():
    d = load(os.path.join(SR, 'out_fig3.json'))
    labs = {'trace + trace-free (first round)': (r'impose angular eqs. (trace + trace-free)', BLUE, 2.2),
            '$\\tau\\tau$ + trace-free': (r'impose $\tau\tau$ + trace-free', ORANGE, 1.6),
            '$\\tau A$ + trace-free': (r'impose $\tau A$ + trace-free', AQUA, 1.6),
            '$\\tau\\tau$ + trace': (r'impose $\tau\tau$ + trace', YELLOW, 1.6)}
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for key, (lab, col, lw) in labs.items():
        e = d['6 ' + key]
        ax.loglog(e['w'], e['R'], color=col, lw=lw, label=lab)
    c = curve(2, 6.0, 'odd')
    from shellgw.junction import Delta_odd
    D = Delta_odd(6.0)
    ww = np.geomspace(0.3, 30, 100)
    ax.loglog(ww, D**2 / (4 * ww**2 + D**2), '--', color=INK, lw=1.1, label=r'$\delta$-barrier law (odd sector, model C)')
    ax.set_xlabel(r'$\omega M$')
    ax.set_ylabel(r'$\mathcal{R}(\omega)$, even parity')
    ax.set_title(r'Frozen domain wall ($R=6M$, $\ell=2$): the answer depends on a choice')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(1e-7, 3)
    save(fig, 'r3_f10_modelA_ambiguity')


# ============================================================ F11 time-domain simulation
def evolve(R=None, l=2, w0=0.35, wid=None, tmax=700.0, h=0.05, x_det=75.0, x0=115.0, xmax=620.0, xbh=-800.0, shell=True):
    """Leapfrog for  Psi_tt = Psi_xx - V Psi - Delta delta(x - x_R) Psi  (odd parity, M = 1) in the global
    tortoise coordinate. R=None: Schwarzschild black hole (absorbing inner boundary far down the throat)."""
    lam = l * (l + 1)
    if R is None:
        xL = xbh
    else:
        s = np.sqrt(1 - 2 / R)
        xR = rstar(R)
        rmin = 0.04 * R
        xL = xR - (R - rmin) / s
    N = int(round((xmax - xL) / h)) + 1
    x = xL + h * np.arange(N)
    Vx = np.zeros(N)
    if R is None:
        r = r_of_x(x)
        Vx = V_RW(r, l)
        iR = None
    else:
        inside = x < xR
        rin = R + s * (x[inside] - xR)
        Vx[inside] = s * s * lam / rin**2
        rout = r_of_x(x[~inside])
        Vx[~inside] = V_RW(rout, l)
        iR = np.argmax(~inside)
        if shell:
            Vx[iR] += s * (1 - s) / R / h
    dt = 0.5 * h
    if wid is None:
        wid = min(max(1.2 * 2 * np.pi / w0, 8), 22)
    F = lambda xi: np.exp(-xi**2 / (2 * wid**2)) * np.cos(w0 * xi)
    u = F(x - x0)
    um = F(x - x0 - dt)
    c2, d2 = (dt / h)**2, dt * dt
    idet = int(round((x_det - xL) / h))
    nsteps = int(tmax / dt)
    ts, sig = [], []
    for n in range(nsteps):
        un = np.empty_like(u)
        un[1:-1] = 2 * u[1:-1] - um[1:-1] + c2 * (u[2:] - 2 * u[1:-1] + u[:-2]) - d2 * Vx[1:-1] * u[1:-1]
        un[-1] = u[-1] - (dt / h) * (u[-1] - u[-2])
        if R is None:
            un[0] = u[0] + (dt / h) * (u[1] - u[0])              # outgoing into the horizon
        else:
            un[0] = 0.0                                          # regular centre (Psi ~ r^{l+1})
        um, u = u, un
        if n % 4 == 0:
            ts.append((n + 1) * dt)
            sig.append(u[idet])
    return np.array(ts), np.array(sig)


def fit_ringdown(t, s, t0, t1):
    """Fit s = A e^{Im(w) t} cos(Re(w) t + phi) on [t0, t1] (local extrema of |s| give the envelope)."""
    from scipy.optimize import curve_fit
    m = (t > t0) & (t < t1)
    tt, ss = t[m], s[m]
    g = lambda t, A, wr, wi, ph: A * np.exp(wi * (t - t0)) * np.cos(wr * (t - t0) + ph)
    best = None
    for wr0 in np.linspace(0.2, 0.9, 15):
        try:
            p, _ = curve_fit(g, tt, ss, p0=[np.abs(ss).max(), wr0, -0.05, 0.0], maxfev=20000)
            res = np.sum((g(tt, *p) - ss)**2)
            if best is None or res < best[1]:
                best = (p, res)
        except RuntimeError:
            pass
    p = best[0]
    return abs(p[1]), p[2]


def fig_timedomain():
    """Broadband Gaussian pulse (width 3M) sent in from x = 115M; detector at x = 75M. The incoming pulse
    passes the detector at t = 40M; anything scattered near the object returns at t ~ 190M."""
    runs = [(None, 'black hole', GRAY, 1.8), (3.0, r'shell $R=3M$', BLUE, 1.0), (2.2, r'shell $R=2.2M$', ORANGE, 1.0)]
    res = {}
    for R, lab, col, lw in runs:
        res[lab] = evolve(R, w0=0.0, wid=3.0, tmax=900.0)
    t, s = res['black hole']
    k = t > 120
    tpk = t[np.argmax(np.abs(s) * k)]
    wr_bh, wi_bh = fit_ringdown(t, s, tpk + 15, tpk + 70)
    t, s = res[r'shell $R=2.2M$']
    wr, wi = fit_ringdown(t, s, 400, 890)
    q22 = modes(2, 2.2, 'odd')[0]['w1']
    OUT['timedomain'] = dict(pulse='gaussian width 3M, x0=115M, detector 75M', fit_R22=[wr, wi], qnm_R22=q22,
                             fit_bh=[wr_bh, wi_bh], qnm_bh=[0.37367168, -0.08896232])
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for R, lab, col, lw in runs:
        t, s = res[lab]
        ax.semilogy(t, np.abs(s) + 1e-16, color=col, lw=lw, label=lab)
    tt = np.linspace(380, 880, 10)
    ax.semilogy(tt, 4e-3 * np.exp(wi * (tt - 380)), '--', color=INK, lw=1.0)
    ax.text(560, 1.2e-2, (r'late-time fit: $\omega M=%.4f%.4fi$' % (wr, wi)) + '\n'
            + (r'QNM (frequency domain): $%.4f%.4fi$' % tuple(q22)), fontsize=8.3)
    ax.set_ylim(1e-9, 3)
    ax.set_xlim(150, 900)
    ax.set_xlabel(r'time at the detector $t/M$')
    ax.set_ylabel(r'$|\Psi|$ at $x=75M$')
    ax.set_title(r'Time-domain simulation: the same pulse ($\ell=2$, odd) hits each object')
    ax.legend(loc='lower left', fontsize=8.5)
    save(fig, 'r3_f11_timedomain')
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    tb, sb = res['black hole']
    t, s = res[r'shell $R=2.2M$']
    ax.plot(tb, sb, color=GRAY, lw=2.0, label='black hole')
    ax.plot(t, s, color=ORANGE, lw=1.0, label=r'shell $R=2.2M$')
    ax.set_xlim(170, 420)
    ax.set_xlabel(r'time at the detector $t/M$')
    ax.set_ylabel(r'$\Psi$ at $x=75M$')
    ax.set_title('Waveform: the black hole rings down, the compact shell keeps ringing')
    ax.legend(fontsize=8.5)
    save(fig, 'r3_f12_waveform')


# ============================================================ F13 multipoles
def fig_multipoles():
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for k, l in enumerate((2, 3, 4)):
        co, ce = curve(l, 3.0, 'odd'), curve(l, 3.0, 'even')
        ax.plot(co['w'], co['R1'], color=SEQ[k + 1], label=r'$\ell=%d$ odd' % l)
        ax.plot(ce['w'], ce['R1'], '--', color=SEQ[k + 1], lw=1.3, label=r'$\ell=%d$ even (model C)' % l)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel(r'$\omega M$')
    ax.set_ylabel(r'$\mathcal{R}(\omega)$')
    ax.set_title(r'Shell at $R=3M$: higher multipoles see a higher barrier')
    ax.legend(fontsize=8.3, ncol=1)
    save(fig, 'r3_f13_multipoles')


# ============================================================ F14 equation of state
def fig_eos():
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for k, G in enumerate((1.6, 2.0, 3.0)):
        c = curve(2, 3.0, 'even', 'C', G)
        ax.semilogy(c['w'], c['R1'], color=[AQUA, ORANGE, VIOLET][k], label=r'even, $\Gamma=%.1f$ ($v_s^2=%.2f$)' % (G, c['vs2']))
    c = curve(2, 3.0, 'odd')
    ax.semilogy(c['w'], c['R1'], ':', color=BLUE, label='odd (independent of $\Gamma$)')
    ax.set_xlim(0, 2)
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel(r'$\omega M$')
    ax.set_ylabel(r'$\mathcal{R}(\omega)$')
    ax.set_title(r'Even parity, $R=3M$, $\ell=2$: dependence on the equation of state')
    ax.legend(fontsize=8.3, loc='upper right')
    save(fig, 'r3_f14_eos')


if __name__ == '__main__':
    fig_multipoles()
    fig_eos()
    fig_potential()
    fig_reflectivity()
    fig_bh_limit()
    fig_qnm_plane()
    fig_matter()
    fig_timescales()
    fig_validation()
    fig_thick()
    fig_modelA()
    fig_timedomain()
    fig_highfreq()
    with open(os.path.join(HERE, 'numbers_r3.json'), 'w') as fh:
        json.dump(OUT, fh, indent=1, default=float)
    print(json.dumps(OUT, indent=1, default=float))
