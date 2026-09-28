"""
h_+ / h_x and the two parities: computations (results cached in ../data/*.json).

  stage A : check of the polarization formula  h_+ - i h_x = (1/2r) sum sqrt((l+2)!/(l-2)!) (Psi_e - i Psi_o) _{-2}Y_lm
            (done symbolically in check_spin_weight.py)
  stage B : black hole: S_even / S_odd vs the Chandrasekhar phase (l+ ...)
  stage C : closed-system reflection amplitudes S_e(w), S_o(w) of shells (l=2), dense adaptive grid
  stage D : helicity-flip cross section  sigma_flip(w) = (pi/w^2) sum_l (2l+1) |S_e - S_o|^2 / 4
  stage E : time domain: reflected waveform of the same incident pulse in the two parities (Fourier synthesis,
            causal treatment of the unstable matter mode), and a leapfrog check of the odd channel.

Physics code (junction matrices, exterior/interior solutions) is a copy of the project code in ../../Third round/code/fr_code
(read only). Run:  python -B parity_study.py [B C D E]
"""
import os
import sys
import json
import time
import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'Third round', 'code', 'fr_code'))
DATA = os.path.join(HERE, '..', 'data')
os.makedirs(DATA, exist_ok=True)
import warnings
warnings.filterwarnings('ignore')
from shellgw.scattering import closed_S               # noqa: E402
from shellgw.blackhole import bh_reflectivity          # noqa: E402

M = 1.0


def vs2(R, G=2.0):
    s = np.sqrt(1 - 2 / R)
    k = (1 - s) / (4 * s)
    return G * k / (1 + k)


def S_shell(w, l, par, R):
    if par == 'odd':
        return complex(closed_S(w, l, 'odd', R))
    return complex(closed_S(w, l, 'even', R, model='C', vs2=vs2(R)))


def S_bh(w, l, par):
    return complex(bh_reflectivity(w, l, par)[0])


def chandra(w, l):
    lam = (l - 1) * l * (l + 1) * (l + 2)
    return (lam + 12j * M * w) / (lam - 12j * M * w)


def dump(name, obj):
    with open(os.path.join(DATA, name), 'w') as fh:
        json.dump(obj, fh)


def load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


# narrow resonances close to the real axis (l = 2) used to seed the frequency grids:
# trapped/cavity modes and the stable matter pair of the fluid shell
SEEDS = {2.2: {'odd': [0.42110968 - 0.03502424j],
               'even': [0.44693900 - 0.04385000j, 0.185919 - 0.000030j]},
         3.0: {'odd': [], 'even': [0.197121 - 0.000310j]},
         6.0: {'odd': [], 'even': [0.089419 - 0.000054j]}}
GROWTH = {2.2: 0.2442920687935753, 3.0: 0.13321, 6.0: 0.040138}     # unstable matter mode  w = i*gamma (l=2)


def grid(R, par, wmin=0.004, wmax=1.6, dw=0.004):
    base = np.arange(wmin, wmax + dw / 2, dw)
    extra = []
    for wp in SEEDS.get(R, {}).get(par, []):
        a, b = wp.real, -wp.imag
        extra.append(a + b * np.tan(np.linspace(-1.55, 1.55, 401)))
    g = np.unique(np.concatenate([base] + extra)) if extra else base
    return g[(g >= wmin) & (g <= wmax)]


def refine(ws, Ss, f, maxpass=6, tol=0.25):
    """Bisect intervals where the phase of S jumps by more than tol."""
    ws, Ss = list(ws), list(Ss)
    for _ in range(maxpass):
        new = []
        for i in range(len(ws) - 1):
            d = abs(np.angle(Ss[i + 1] / Ss[i]))
            if d > tol and ws[i + 1] - ws[i] > 1e-7:
                new.append(0.5 * (ws[i] + ws[i + 1]))
        if not new:
            break
        for w in new:
            ws.append(w)
            Ss.append(f(w))
        order = np.argsort(ws)
        ws = list(np.array(ws)[order])
        Ss = list(np.array(Ss)[order])
    return np.array(ws), np.array(Ss)


# ---------------------------------------------------------------------------------------------- stage B
def stage_B():
    out = {}
    ws = np.geomspace(0.02, 1.6, 90)
    for l in (2, 3, 4):
        for par in ('odd', 'even'):
            out['%d %s' % (l, par)] = [[S.real, S.imag] for S in (S_bh(w, l, par) for w in ws)]
    out['w'] = list(ws)
    dump('bh_S.json', out)
    err = []
    for l in (2, 3, 4):
        So = np.array([complex(*x) for x in out['%d odd' % l]])
        Se = np.array([complex(*x) for x in out['%d even' % l]])
        rat = Se / So
        err.append(np.max(np.abs(rat - chandra(ws, l))[ws < 0.8]))
    print('stage B: max |S_e/S_o - Chandrasekhar| (w<0.8) for l=2,3,4:', err)


# ---------------------------------------------------------------------------------------------- stage C
def stage_C():
    out = {}
    for R in (2.2, 3.0, 6.0):
        for par in ('odd', 'even'):
            t0 = time.time()
            ws = grid(R, par)
            Ss = np.array([S_shell(w, 2, par, R) for w in ws])
            ws, Ss = refine(ws, Ss, lambda w: S_shell(w, 2, par, R))
            out['%g %s' % (R, par)] = dict(w=list(ws), S=[[s.real, s.imag] for s in Ss])
            print('stage C', R, par, len(ws), 'points, %.0f s' % (time.time() - t0), 'max||S|-1| = %.1e' % np.max(np.abs(np.abs(Ss) - 1)))
    # black hole on a comparable grid (l = 2)
    ws = np.arange(0.004, 1.6 + 1e-9, 0.004)
    for par in ('odd', 'even'):
        Ss = np.array([S_bh(w, 2, par) for w in ws])
        out['bh %s' % par] = dict(w=list(ws), S=[[s.real, s.imag] for s in Ss])
    dump('shell_S_l2.json', out)


# ---------------------------------------------------------------------------------------------- stage D
def stage_D():
    ws = np.geomspace(0.02, 1.2, 36)
    out = {'w': list(ws)}
    for R in ('bh', 3.0, 6.0, 2.2):
        sig, sig_rel, lmaxs = [], [], []
        for w in ws:
            rc = 3 * np.sqrt(3) if R == 'bh' else max(R, 3 * np.sqrt(3))
            lmax = max(8, int(2.2 * w * rc) + 7)
            tot, terms = 0.0, []
            for l in range(2, lmax + 1):
                if R == 'bh':
                    So, Se = S_bh(w, l, 'odd'), S_bh(w, l, 'even')
                else:
                    So, Se = S_shell(w, l, 'odd', R), S_shell(w, l, 'even', R)
                t = (2 * l + 1) * abs(Se - So)**2 / 4
                terms.append(t)
                tot += t
            sig.append(np.pi / w**2 * tot)
            sig_rel.append(terms[-1] / max(tot, 1e-300))
            lmaxs.append(lmax)
        out[str(R)] = dict(sigma=sig, last_term_rel=sig_rel, lmax=lmaxs)
        print('stage D', R, 'sigma_flip(w->0) = %.4f M^2' % sig[0], ' (4 pi/3 = %.4f)' % (4 * np.pi / 3),
              'max rel. last term %.1e' % max(sig_rel))
    # analytic black-hole value with the numerical reflectivities (check of the formula)
    ana = []
    for w in ws:
        tot = 0.0
        for l in range(2, 12):
            lam = (l - 1) * l * (l + 1) * (l + 2)
            Rl = abs(S_bh(w, l, 'odd'))**2
            tot += (2 * l + 1) * Rl * np.sin(np.arctan(12 * w / lam))**2
        ana.append(np.pi / w**2 * tot)
    out['bh_analytic'] = ana
    dump('sigma_flip.json', out)


# ---------------------------------------------------------------------------------------------- stage E
def pulse_ft(w, sigma, w0, v0):
    """Fourier transform  I~(w) = int I(v) e^{i w v} dv  of  I(v) = exp(-(v-v0)^2/2 sigma^2) cos(w0 (v-v0))."""
    g = lambda x: np.exp(-sigma**2 * x**2 / 2)
    return np.exp(1j * w * v0) * sigma * np.sqrt(2 * np.pi) / 2 * (g(w - w0) + g(w + w0))


def filon(ws, F, us):
    """(1/pi) Re int_0^inf F(w) e^{-i w u} dw with F piecewise linear on the (non-uniform) grid ws."""
    ws = np.asarray(ws)
    F = np.asarray(F)
    out = np.zeros(len(us))
    a, b = ws[:-1], ws[1:]
    h = b - a
    Fa, Fb = F[:-1], F[1:]
    for j, u in enumerate(us):
        k = -1j * u
        kh = k * h
        ea, eb = np.exp(k * a), np.exp(k * b)
        small = np.abs(kh) < 1e-4
        with np.errstate(divide='ignore', invalid='ignore'):
            I0 = np.where(small, h * 0.5 * (ea + eb), (eb - ea) / k)
            I1 = np.where(small, h * h * (ea / 6 + eb / 3), h * eb / k - (eb - ea) / k**2)
        val = np.sum(Fa * I0 + (Fb - Fa) / h * I1)
        out[j] = val.real / np.pi
    return out


def synth(key, sigma=8.0, w0=0.5, v0=0.0, us=None, r_det=None, r_src=None):
    d = load('shell_S_l2.json')[key]
    ws = np.array(d['w'])
    S = np.array([complex(*x) for x in d['S']])
    m = (ws >= 0.004) & (ws <= 1.6)
    ws, S = ws[m], S[m]
    F = S * pulse_ft(ws, sigma, w0, 0.0)       # the factor e^{i w v0} is applied exactly, as a shift of u
    if r_det is not None:                      # near-zone factor of the outgoing solution at a finite radius
        from shellgw.exterior import up_asymptotic, rstar_c
        par = key.split()[1]
        fac = np.ones(len(ws), complex)
        for k, w in enumerate(ws):
            if w * r_det > 3.5:
                P = up_asymptotic(r_det, 2, par, w)[0]
                fac[k] = P / np.exp(1j * w * rstar_c(r_det))
        F = F * fac
    if r_src is not None:                      # initial data F(x - x0) at finite radius: amplitude of the exact ingoing wave
        from shellgw.exterior import up_asymptotic, rstar_c
        par = key.split()[1]
        fac = np.ones(len(ws), complex)
        for k, w in enumerate(ws):
            if w * r_src > 3.5:
                P = up_asymptotic(r_src, 2, par, w)[0]
                fac[k] = np.conj(P / np.exp(1j * w * rstar_c(r_src)))
        F = F / fac
    return filon(ws, F, np.asarray(us) - v0)


def stage_E(sigma=8.0, w0=0.5):
    from residue import residue
    out = {'u': None, 'pulse': dict(sigma=sigma, w0=w0)}
    us = np.concatenate([np.arange(-200, 400, 0.5), np.arange(400, 3000, 2.0)])
    out['u'] = list(us)
    R0 = {2.2: 30.0, 3.0: 45.0, 6.0: 110.0}
    for key in ('bh odd', 'bh even', '2.2 odd', '2.2 even', '3 odd', '3 even', '6 odd', '6 even'):
        t0 = time.time()
        O = synth(key, sigma, w0, us=us)
        rec = dict(real_axis=list(O))
        if key.endswith('even') and not key.startswith('bh'):
            R = float(key.split()[0])
            wp, rS, err = residue(R, GROWTH[R], R0[R])
            amp = -1j * rS * pulse_ft(wp, sigma, w0, 0.0)          # growing-mode term  amp * e^{-i wp u}
            grow = (amp * np.exp(-1j * wp * us)).real
            rec.update(pole=[wp.real, wp.imag], residue=[rS.real, rS.imag], amp=[amp.real, amp.imag],
                       gamma=wp.imag, growth=list(grow), causal=list(O + grow),
                       u_equal=float(np.log(1.0 / abs(amp)) / wp.imag))
            print('   pole %s residue %s growing-mode amplitude at u=0: %.2e, reaches 1 at u=%.0f M'
                  % (np.round(wp, 7), np.round(rS, 8), abs(amp), rec['u_equal']))
        else:
            rec['causal'] = list(O)
        out[key] = rec
        print('stage E', key, '%.0f s' % (time.time() - t0))
    # leapfrog check of the odd channel (R = 3M, 2.2M): pulse centred at v0 = 115 M, detector at x = 75 M
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'Third round', 'code'))
    import make_figures_r3 as mf
    x0, xd = 450.0, 350.0                      # source and detector far out (near-zone factors applied)
    r_det, r_src = float(mf.r_of_x(xd)), float(mf.r_of_x(x0))
    chk = {}
    for R in (3.0, 2.2):
        t, s = mf.evolve(R, w0=w0, wid=sigma, tmax=x0 + xd + 320.0, x0=x0, x_det=xd)
        uu = t - xd - x0                           # retarded time measured from the pulse centre
        m = (uu > -60) & (uu < 320)
        Of = synth('%g odd' % R, sigma, w0, v0=0.0, us=uu[m], r_det=r_det, r_src=r_src)
        chk[str(R)] = dict(u=list(uu[m]), leapfrog=list(s[m]), synth=list(Of),
                           max_diff=float(np.max(np.abs(s[m] - Of))), max_sig=float(np.max(np.abs(Of))))
        print('leapfrog check R=%g: max|diff| = %.2e of max %.2e' % (R, chk[str(R)]['max_diff'], chk[str(R)]['max_sig']))
    out['leapfrog'] = chk
    dump('timedomain.json', out)


if __name__ == '__main__':
    stages = sys.argv[1:] or ['B', 'C', 'D', 'E']
    for st in stages:
        globals()['stage_' + st]()
