"""
Production computations (scattering curves, QNM catalogues, QNM tracks, validation
tables).  All functions are top-level so they can run in a multiprocessing pool;
results are cached as JSON in ../../results.
"""
import json
import os
import numpy as np
from .background import kappa_static, dynamical_frequency, Rddot_domain_wall
from .scattering import barrier_RT, coeffs_method2, closed_S
from .qnm import scan_fast, refine, find_qnm
from .blackhole import leaver_qnm, bh_reflectivity
from .junction import constraint_residual

RESDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
GAMMA_FID = 2.0          # fiducial adiabatic index (Pitre, Schneider & Poisson 2026)


def vs2_of(R, Gamma=GAMMA_FID, M=1.0):
    """delta p/delta sigma = Gamma p/(sigma + p) for the polytropic shell p = K n^Gamma
    (Pitre et al. 2026 Eqs. 2.4-2.6: dp = Gamma p/(mu+p) dmu, mu = our sigma)."""
    k = kappa_static(R, M)
    return Gamma * k / (1 + k)


def cfg_label(parity, model):
    if parity == 'odd':
        return 'odd'
    return 'even-' + model


def _save(name, obj):
    os.makedirs(RESDIR, exist_ok=True)
    with open(os.path.join(RESDIR, name), 'w') as fh:
        json.dump(obj, fh, indent=1)


def _load(name):
    p = os.path.join(RESDIR, name)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def cplx(z):
    return [float(np.real(z)), float(np.imag(z))]


# ------------------------------------------------------------------ scattering
def rt_curve(args):
    """R(w), T(w) of the barrier-plus-shell problem for one configuration.
    args = (parity, model, l, R, Gamma, wgrid, with_method2)"""
    parity, model, l, R, Gamma, wgrid, m2 = args
    vs2 = vs2_of(R, Gamma) if (parity == 'even' and model == 'C') else 0.0
    out = dict(parity=parity, model=model, l=l, R=R, Gamma=Gamma, vs2=vs2, w=list(wgrid),
               R1=[], T1=[], R2=[], T2=[], Rbw=[], S_phase=[], constr=[])
    for w in wgrid:
        R1, T1 = barrier_RT(w, l, parity, R, 1.0, model, vs2, method=1)
        out['R1'].append(R1)
        out['T1'].append(T1)
        if m2:
            Bin, Bout, est = coeffs_method2(w, l, parity, R, 1.0, model, vs2, 'ingoing', window=True)
            out['R2'].append(abs(Bout / Bin)**2)
            out['T2'].append(1.0 / abs(Bin)**2)
            out['Rbw'].append(est**2)
            S = closed_S(w, l, parity, R, 1.0, model, vs2)
            out['S_phase'].append(float(np.angle(S)))
        if parity == 'even' and model == 'A':
            out['constr'].append(constraint_residual(w, l, R, 1.0, 'A'))
    return out


def bh_curve(args):
    l, parity, wgrid = args
    Rb, Tb = [], []
    for w in wgrid:
        ra, ta = bh_reflectivity(w, l, parity)
        Rb.append(abs(ra)**2)
        Tb.append(abs(ta)**2)
    return dict(l=l, parity=parity, w=list(wgrid), R=Rb, T=Tb)


# ------------------------------------------------------------------ QNM catalogues
def pn_matter_seeds(l, Gamma):
    """Post-Newtonian matter-mode parameters varsigma_0 (Pitre et al. 2026 Eq. 7.8)."""
    L = l * l + l
    a = ((L + 4) * Gamma - (L + 6)) / 8.0
    disc = (L + 4)**2 * Gamma**2 + 2 * (2 * l**5 - 3 * l**4 - 40 * l**3 - 33 * l**2 - 46 * l - 24) * Gamma / (2 * l + 1) \
        + (2 * l**5 - 11 * l**4 + 60 * l**3 + 53 * l**2 + 52 * l + 36) / (2 * l + 1)
    sp_ = a + np.sqrt(disc) / 8.0
    sm_ = a - np.sqrt(disc) / 8.0
    return np.sqrt(sp_ + 0j), np.sqrt(sm_ + 0j)       # real (stable pair), imaginary (unstable)


def qnm_catalogue(args):
    """All QNMs of one configuration in a window of the complex plane.
    args = (parity, model, l, R, Gamma)"""
    parity, model, l, R, Gamma = args
    vs2 = vs2_of(R, Gamma) if (parity == 'even' and model == 'C') else 0.0
    cands = []
    wr, wi, G, c = scan_fast(l, parity, R, 1.0, model, vs2, re=(0.02, 2.6), im=(-1.3, -0.002),
                             nre=170, nim=90)
    cands += c
    if R < 3.2:          # long-lived trapped modes near the real axis
        wr2, wi2, G2, c2 = scan_fast(l, parity, R, 1.0, model, vs2, re=(0.02, 1.2), im=(-0.03, -1e-6),
                                     nre=240, nim=40)
        cands += c2
    modes = []
    for o in refine(cands, l, parity, R, 1.0, model, vs2):
        modes.append(dict(kind='wave', w1=cplx(o['w1']), w2=cplx(o['w2']), diff=o['diff'], res=o['res1']))
    if parity == 'even' and model == 'C':
        sR, sI = pn_matter_seeds(l, Gamma)
        sc = np.sqrt(1.0 / R**3)
        for g, kind in ((sR.real * sc - 1e-4j * sc, 'matter-stable'), (1j * abs(sI) * sc, 'matter-unstable'),
                        (-1j * abs(sI) * sc, 'matter-imag-damped')):
            w1, r1, ok = find_qnm(g, l, parity, R, 1.0, model, vs2, method=1)
            if ok and r1 < 1e-8:
                w2, r2, ok2 = find_qnm(w1, l, parity, R, 1.0, model, vs2, method=2)
                modes.append(dict(kind=kind, w1=cplx(w1), w2=cplx(w2), diff=abs(w1 - w2), res=r1))
    # wave-mode ordering label n (by increasing Re w, as in Pitre et al. 2026 Figs. 3-4)
    wave = sorted([m for m in modes if m['kind'] == 'wave'], key=lambda m: m['w1'][0])
    for n, m in enumerate(wave):
        m['n'] = n
    for m in modes:
        m.setdefault('n', 0)
    return dict(parity=parity, model=model, l=l, R=R, Gamma=Gamma, vs2=vs2, modes=modes)


def bh_qnms(lmax=4, nmax=3):
    """Schwarzschild QNMs (Leaver) with WKB (orders 1, 3, 6) comparisons."""
    from .potentials import peak_derivatives
    from .wkb import wkb_qnm
    out = []
    for l in range(2, lmax + 1):
        r0, d = peak_derivatives(l, 'odd', 12)
        for n in range(nmax + 1):
            g = wkb_qnm(d, n, 6)
            w, err = leaver_qnm(l, g, inv=n)
            row = dict(l=l, n=n, leaver=cplx(w), cf_err=err)
            for order in (1, 3, 6):
                row['wkb%d' % order] = cplx(wkb_qnm(d, n, order))
            out.append(row)
    return out


# ------------------------------------------------------------------ tracks vs R/M
def track(args):
    """Follow one QNM by continuation in R.  args = (parity, model, l, Gamma, R_start, w_start, Rgrid)"""
    parity, model, l, Gamma, R0, w0, Rgrid = args
    pts = []
    w_prev, w_prev2, R_prev, R_prev2 = w0, None, R0, None
    for R in Rgrid:
        vs2 = vs2_of(R, Gamma) if (parity == 'even' and model == 'C') else 0.0
        if w_prev2 is not None:
            g = w_prev + (w_prev - w_prev2) * (R - R_prev) / (R_prev - R_prev2)
        else:
            g = w_prev
        w, res, ok = find_qnm(g, l, parity, R, 1.0, model, vs2, method=1)
        if not (ok and res < 1e-8) or abs(w - g) > 0.3 * abs(g) + 0.05:
            break
        pts.append(dict(R=R, w=cplx(w)))
        w_prev2, R_prev2 = w_prev, R_prev
        w_prev, R_prev = w, R
    return dict(parity=parity, model=model, l=l, Gamma=Gamma, start=cplx(w0), R0=R0, pts=pts)
