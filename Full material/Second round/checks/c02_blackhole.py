"""
CHECK 02 -- Schwarzschild black hole: QNMs, reflectivity, WKB (independent code).

(a) Leaver (1985) continued fraction, implemented from scratch in mpmath
    (units 2M = 1, s = 2), with the n-th inversion, 30 digits.
(b) Black-hole reflectivity |A_out/A_in|^2 (ingoing at the horizon), obtained by
    integrating in the tortoise coordinate x = r* (LSODA on the real 4-vector, not DOP853) from x = -80M
    with Psi = e^{-i w x}, and projecting at large r on an asymptotic series whose
    recursion is re-derived and verified symbolically below.
(c) Third-order WKB from the explicit Iyer & Will (1987) Eq. (1.5) formula
    (not from the hypervirial construction used in the first round).
"""
import json
import numpy as np
import mpmath as mp
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.special import lambertw

mp.mp.dps = 30
OUT = {}

# ------------------------------------------------------------------ (a) Leaver
def leaver(w, l, inv, N=1200):
    w = mp.mpc(w)
    rho = -1j * w                      # 2M = 1, Leaver's rho = -i w (w in units 1/(2M))
    a = lambda n: n**2 + (2*rho + 2)*n + 2*rho + 1
    b = lambda n: -(2*n**2 + (8*rho + 2)*n + 8*rho**2 + 4*rho + l*(l + 1) - 3)   # s^2-1 = 3
    c = lambda n: n**2 + 4*rho*n + 4*rho**2 - 4
    tail = mp.mpc(0)
    for n in range(N, inv, -1):
        tail = a(n - 1)*c(n)/(b(n) - tail)
    head = mp.mpc(0)
    if inv > 0:
        h = b(0)
        for n in range(1, inv):
            h = b(n) - a(n - 1)*c(n)/h
        head = a(inv - 1)*c(inv)/h
    return b(inv) - head - tail


BCS09 = {(2, 0): 0.373672-0.088962j, (2, 1): 0.346711-0.273915j, (2, 2): 0.301053-0.478277j,
         (2, 3): 0.251505-0.705148j, (3, 0): 0.599443-0.092703j, (3, 1): 0.582644-0.281298j,
         (3, 2): 0.551685-0.479093j, (3, 3): 0.511962-0.690337j, (4, 0): 0.809178-0.094164j,
         (4, 1): 0.796632-0.284334j, (4, 2): 0.772710-0.479908j, (4, 3): 0.739837-0.683924j}
first = {}
with open('../../First round/results/validation.json') as fh:
    V1 = json.load(fh)
for b in V1['bh_qnm']:
    first[(b['l'], b['n'])] = complex(*b['leaver'])

qnm = {}
print('(a) Leaver QNMs  (wM)            |mine - BCS09 table|   |mine - first round|')
for (l, n), g in BCS09.items():
    root = mp.findroot(lambda x: leaver(x, l, n), mp.mpc(2*g))      # w in 1/(2M)
    root2 = mp.findroot(lambda x: leaver(x, l, n, N=2400), root)
    wM = complex(root2)/2
    qnm[(l, n)] = wM
    print('  l=%d n=%d  %.10f%+.10fi   %.1e   %.1e   (depth change %.1e)'
          % (l, n, wM.real, wM.imag, abs(wM - g), abs(wM - first[(l, n)]), abs(complex(root2 - root))/2))
OUT['leaver'] = {'%d,%d' % k: [v.real, v.imag] for k, v in qnm.items()}

# ------------------------------------------------------------------ (b) reflectivity
# asymptotic recursion: re-derive by substituting Psi = e^{i w r*} sum c_k r^-k in the RW equation
r, w_, M_, lam_ = sp.symbols('r omega M lambda')
K = 6
cs = sp.symbols('c0:%d' % (K + 2))
u = sum(cs[k]*r**(-k) for k in range(K + 2))
f = 1 - 2*M_/r
# with Psi = e^{i w r*} u the RW equation divided by f reads
#   f u'' + (f' + 2 i w) u' - (lam/r^2 - 6M/r^3) u = 0
expr = sp.expand((f*sp.diff(u, r, 2) + (sp.diff(f, r) + 2*sp.I*w_)*sp.diff(u, r)
                  - (lam_/r**2 - 6*M_/r**3)*u)*r**(K + 4))
claim_ok = True
for k in range(1, K):
    coeff = expr.coeff(r, K + 4 - (k + 2))                     # coefficient of r^-(k+2)
    rec = -(2*sp.I*w_*(k + 1)*cs[k + 1] - ((k*(k + 1) - lam_)*cs[k] - 2*M_*(k**2 - 4)*cs[k - 1]))
    d = sp.simplify(coeff - rec)
    claim_ok &= (d == 0)
    print('   k=%d: coefficient of r^-%d minus claimed recursion =' % (k, k + 2), d)
print('   first-round asymptotic recursion (derivation Eq. 5.2) reproduced:', claim_ok)


def asym_series(rr, l, w, M=1.0, kmax=40):
    lam = l*(l + 1)
    c = [1.0 + 0j, ((0 - lam)*1.0)/(2j*w)]
    for k in range(1, kmax):
        c.append(((k*(k + 1) - lam)*c[k] - 2*M*(k*k - 4)*c[k - 1])/(2j*w*(k + 1)))
    terms = [c[k]*rr**(-k) for k in range(kmax + 1)]
    mags = np.abs(terms)
    stop = kmax
    for k in range(3, kmax):
        if mags[k] > mags[k - 1] and mags[k - 1] > 0:
            stop = k - 1
            break
    uu = sum(terms[:stop])
    du = sum(-k*c[k]*rr**(-k - 1) for k in range(stop))
    xs = rr + 2*M*np.log(rr/(2*M) - 1)
    ph = np.exp(1j*w*xs)
    f = 1 - 2*M/rr
    return ph*uu, ph*(1j*w*uu + f*du)           # (Psi, dPsi/dr*)


def r_of_x(x, M=1.0):
    return 2*M*(1 + np.real(lambertw(np.exp(x/(2*M) - 1))))


def Vrw(rr, l, M=1.0):
    return (1 - 2*M/rr)*(l*(l + 1)/rr**2 - 6*M/rr**3)


def bh_R(w, l, M=1.0, x0=-80.0, rfar=None):
    if rfar is None:
        rfar = max(300.0, 80/w)
    xf = rfar + 2*M*np.log(rfar/(2*M) - 1)

    def rhs(x, y):                         # real 4-vector (Re Psi, Re Psi', Im Psi, Im Psi')
        rr = r_of_x(x)
        q = Vrw(rr, l) - w*w
        return [y[1], q*y[0], y[3], q*y[2]]
    z0 = np.exp(-1j*w*x0)
    y0 = [z0.real, (-1j*w*z0).real, z0.imag, (-1j*w*z0).imag]
    sol = solve_ivp(rhs, (x0, xf), y0, method='LSODA', rtol=1e-11, atol=1e-13)
    P = sol.y[0, -1] + 1j*sol.y[2, -1]
    dP = sol.y[1, -1] + 1j*sol.y[3, -1]
    Pu, dPu = asym_series(rfar, l, w)
    A = np.array([[np.conj(Pu), Pu], [np.conj(dPu), dPu]])
    Bin, Bout = np.linalg.solve(A, [P, dP])
    return abs(Bout/Bin)**2, 1/abs(Bin)**2


print('\n(b) BH reflectivity, l = 2 (odd):  (2Mw)^2   R_mine   R+T-1   R_first_round')
refl = []
for row in V1['vishveshwara']:
    w = np.sqrt(row['k2'])/2
    Rm, Tm = bh_R(w, 2)
    refl.append(dict(k2=row['k2'], R=Rm, RT=Rm + Tm - 1, R1=row['R_odd'], R1even=row['R_even']))
    print('   %.3f  %.6e  %.1e  %.6e  (first-round even: %.6e)' % (row['k2'], Rm, Rm + Tm - 1, row['R_odd'], row['R_even']))
OUT['bh_reflectivity'] = refl

# ------------------------------------------------------------------ (c) Iyer-Will 3rd-order WKB
rs = sp.Symbol('r', positive=True)


def wkb3(l, n, M=1):
    f_ = 1 - 2*M/rs
    V = f_*(l*(l + 1)/rs**2 - sp.Rational(6*M)/rs**3)
    ders = [V]
    for _ in range(6):
        ders.append(sp.simplify(f_*sp.diff(ders[-1], rs)))
    r0 = sp.nsolve(sp.diff(V, rs), rs, 3.0)
    d = [complex(sp.N(dd.subs(rs, r0), 30)) for dd in ders]
    V0, V2, V3, V4, V5, V6 = d[0], d[2], d[3], d[4], d[5], d[6]
    a = n + 0.5
    s2 = np.sqrt(-2*V2 + 0j)
    Lam = (1/s2)*((1/8)*(V4/V2)*(0.25 + a*a) - (1/288)*(V3/V2)**2*(7 + 60*a*a))
    Om = (1/(-2*V2))*((5/6912)*(V3/V2)**4*(77 + 188*a*a) - (1/384)*(V3**2*V4/V2**3)*(51 + 100*a*a)
                      + (1/2304)*(V4/V2)**2*(67 + 68*a*a) + (1/288)*(V3*V5/V2**2)*(19 + 28*a*a)
                      - (1/288)*(V6/V2)*(5 + 4*a*a))
    w2 = V0 + s2*Lam - 1j*a*s2*(1 + Om)
    ww = np.sqrt(w2)
    return ww if ww.real > 0 else -ww


print('\n(c) WKB 3rd order (explicit Iyer-Will formula) vs first round (hypervirial):')
wk = {}
for b in V1['bh_qnm']:
    w3 = wkb3(b['l'], b['n'])
    wk['%d,%d' % (b['l'], b['n'])] = [w3.real, w3.imag]
    print('   l=%d n=%d  IW: %.5f%+.5fi   first round: %.5f%+.5fi   diff %.1e ;  rel.err vs Leaver: %.1e'
          % (b['l'], b['n'], w3.real, w3.imag, b['wkb3'][0], b['wkb3'][1], abs(w3 - complex(*b['wkb3'])),
             abs(w3 - qnm[(b['l'], b['n'])])/abs(qnm[(b['l'], b['n'])])))
OUT['wkb3_IW'] = wk
# relative vs absolute errors quoted in the derivation (Sec. 6.1: 7e-2, 1.5e-3, 2.3e-4 for l=2,n=0)
b = [x for x in V1['bh_qnm'] if x['l'] == 2 and x['n'] == 0][0]
L = qnm[(2, 0)]
for order in (1, 3, 6):
    z = complex(*b['wkb%d' % order])
    print('   l=2,n=0 WKB%d: absolute error %.2e, relative error %.2e' % (order, abs(z - L), abs(z - L)/abs(L)))
with open('out_c02.json', 'w') as fh:
    json.dump(OUT, fh, indent=1)
