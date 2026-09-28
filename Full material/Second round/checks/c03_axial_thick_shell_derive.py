"""
CHECK 03a -- odd-parity (axial) perturbations of a general static, spherically symmetric
anisotropic-fluid spacetime, derived from scratch (my own brute-force linearisation;
no first-round code used).

Background: ds^2 = -e^{2 Phi(r)} dt^2 + dr^2/(1 - 2 m(r)/r) + r^2 dOmega^2,
T_mu nu = (rho + p_t) u_mu u_nu + p_t g_mu nu + (p_r - p_t) s_mu s_nu,
u = e^{-Phi} d_t,  s_mu = e^{Lambda} delta^r_mu  (normal to the matter layers r = const).

Axial perturbation (RW gauge, l = 2 explicit, m = 0):
  h_{t phi} = h0(r) e^{-i w t} sin(th) dY/dth,  h_{r phi} = h1(r) e^{-i w t} sin(th) dY/dth,
  delta u_phi = U(r) e^{-i w t} sin(th) dY/dth,   delta rho = delta p = 0, delta s_mu = 0.

Goal: find the master equation for X = e^{Phi - Lambda} h1 / r in the tortoise
coordinate dr* = e^{Lambda - Phi} dr, and its potential V.  The thin-shell limit of V
then gives an INDEPENDENT prediction of the odd junction condition.
"""
import sympy as sp

t, r, th, ph = sp.symbols('t r theta phi', real=True)
w, eps = sp.symbols('omega epsilon')
l = 2
lam = l*(l + 1)
Phi = sp.Function('Phi')(r)
m = sp.Function('m')(r)
rho = sp.Function('rho')(r)
pr = sp.Function('p_r')(r)
pt = sp.Function('p_t')(r)
h0 = sp.Function('h0')(r)
h1 = sp.Function('h1')(r)
U = sp.Function('U')(r)
Y = sp.legendre(l, sp.cos(th))
E = sp.exp(-sp.I*w*t)
A = sp.exp(2*Phi)
B = 1/(1 - 2*m/r)
X = [t, r, th, ph]
g = sp.diag(-A, B, r**2, r**2*sp.sin(th)**2)
ang = sp.sin(th)*sp.diff(Y, th)
g[0, 3] = g[3, 0] = eps*h0*E*ang
g[1, 3] = g[3, 1] = eps*h1*E*ang
ginv = g.inv()
# expand inverse to first order
ginv = ginv.applyfunc(lambda e: sp.series(e, eps, 0, 2).removeO())


def christ(g, gi):
    G = [[[0]*4 for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b in range(4):
            for c in range(b, 4):
                G[a][b][c] = G[a][c][b] = sp.expand(sum(gi[a, d]*(sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b])
                                                                 - sp.diff(g[b, c], X[d])) for d in range(4))/2)
    return G


def trunc(e):
    return sp.expand(e).subs(eps**2, 0).subs(eps**3, 0)


G = christ(g, ginv)
G = [[[trunc(G[a][b][c]) for c in range(4)] for b in range(4)] for a in range(4)]


def ricci(b, d):
    s = 0
    for a in range(4):
        s += sp.diff(G[a][b][d], X[a]) - sp.diff(G[a][b][a], X[d])
        for e in range(4):
            s += G[a][a][e]*G[e][b][d] - G[a][d][e]*G[e][b][a]
    return trunc(s)


# only the components we need; first order in eps
Ric = {(b, d): ricci(b, d) for (b, d) in [(0, 3), (1, 3), (2, 3)]}
# Ricci scalar is unperturbed at first order for axial perturbations -> use background scalar
# background scalar:
g0 = g.subs(eps, 0)
g0i = g0.inv()
G0 = christ(g0, g0i)
Rs0 = 0
for b in range(4):
    for d in range(4):
        if g0i[b, d] == 0:
            continue
        s = 0
        for a in range(4):
            s += sp.diff(G0[a][b][d], X[a]) - sp.diff(G0[a][b][a], X[d])
            for e in range(4):
                s += G0[a][a][e]*G0[e][b][d] - G0[a][d][e]*G0[e][b][a]
        Rs0 += g0i[b, d]*s
Rs0 = sp.simplify(Rs0)
# stress tensor, first order
ut = -sp.exp(Phi)
u_ = [ut, 0, 0, eps*U*E*ang]
s_ = [0, sp.sqrt(B), 0, 0]
T = sp.zeros(4, 4)
for a in range(4):
    for b in range(4):
        T[a, b] = (rho + pt)*u_[a]*u_[b] + pt*g[a, b] + (pr - pt)*s_[a]*s_[b]
eqs = {}
for (a, b) in [(0, 3), (1, 3), (2, 3)]:
    Ein = Ric[(a, b)] - g[a, b]*Rs0/2
    e = sp.diff(trunc(Ein - 8*sp.pi*T[a, b]), eps).subs(eps, 0)
    e = sp.simplify(e/E)
    # remove angular dependence: divide by the appropriate angular function and check
    eqs[(a, b)] = e
    print('component', (a, b), 'computed', flush=True)

# angular reduction
angf = {(0, 3): ang, (1, 3): ang, (2, 3): sp.sin(th)*(sp.diff(Y, th, 2) - sp.cos(th)/sp.sin(th)*sp.diff(Y, th))}
red = {}
for k, e in eqs.items():
    q = sp.simplify(e/angf[k])
    assert not q.has(th), (k, q)
    red[k] = q
    print(k, ':', sp.simplify(q), flush=True)

# background field equations to eliminate m', Phi', p_t
bg = {sp.Derivative(m, r): 4*sp.pi*r**2*rho}
Phip = (m + 4*sp.pi*r**3*pr)/(r*(r - 2*m))
pt_expr = pr + r*(sp.Derivative(pr, r) + (rho + pr)*Phip)/2       # anisotropic TOV


def bgsub(e):
    e = e.subs(pt, pt_expr).doit()
    e = e.subs(sp.Derivative(Phi, (r, 2)), sp.diff(Phip, r))
    e = e.subs(sp.Derivative(Phi, r), Phip)
    e = e.subs(sp.Derivative(m, (r, 2)), sp.diff(4*sp.pi*r**2*rho, r)).subs(bg)
    e = e.subs(sp.Derivative(Phi, r), Phip).subs(bg)
    return sp.simplify(e)


# Solve (theta phi) eq for h0 and (r phi) for relation; build X equation
e23 = bgsub(red[(2, 3)])
e13 = bgsub(red[(1, 3)])
print('theta-phi:', e23)
print('r-phi   :', e13)
h0sol = sp.solve(e23, h0)
print('h0 from theta-phi eq ->', h0sol)
# alternative: theta-phi eq usually involves h0 and h1' ; solve for h0'? handle generically
Xf = sp.Function('X')(r)
Lam = sp.log(sp.sqrt(B))
h1_of_X = Xf*r*sp.exp(Lam - Phi)
# eliminate h0 using theta-phi eq (solve for h0), substitute in r-phi eq
sol0 = sp.solve(e23, h0)[0] if h0sol else None
if sol0 is None:
    sol0p = sp.solve(e23, sp.Derivative(h0, r))[0]
    raise SystemExit('theta-phi eq does not give h0 algebraically; got h0 prime = %s' % sol0p)
e = e13.subs(h0, sol0).doit()
e = e.subs(h1, h1_of_X).doit()
e = bgsub(e)
# write as X_{r*r*} + (w^2 - V) X = 0 with d/dr* = e^{Phi-Lambda} d/dr
Xp, Xpp = sp.symbols('Xp Xpp')
e = e.subs(sp.Derivative(Xf, (r, 2)), Xpp).subs(sp.Derivative(Xf, r), Xp).subs(Xf, sp.Symbol('X0'))
e = sp.expand(sp.simplify(e))
cXpp = sp.simplify(e.coeff(Xpp))
cXp = sp.simplify(e.coeff(Xp))
cX = sp.simplify(e.coeff(sp.Symbol('X0')))
# tortoise: X_{r*r*} = e^{2(Phi-Lam)} [X'' + (Phi' - Lam') X']
F = sp.exp(2*(Phi - Lam))
dlog = bgsub(sp.diff(Phi - Lam, r))
print('check first-derivative structure (should be 0):', sp.simplify(cXp/cXpp - dlog))
Vfound = sp.simplify(bgsub(-(cX/cXpp)*F - w**2)) if True else None
# The equation is cXpp X'' + cXp X' + cX X = 0 -> dividing: X'' + dlog X' + (cX/cXpp) X = 0
# -> e^{-2(Phi-Lam)} X_{r*r*} + (cX/cXpp) X = 0 -> X_{r*r*} + F (cX/cXpp) X = 0 = X_** + (w^2 - V) X
Vfound = sp.simplify(bgsub(w**2 - F*cX/cXpp))
Vclaim = sp.exp(2*Phi)*(lam/r**2 - 6*m/r**3 + 4*sp.pi*(rho - pr))
print('V found =', Vfound)
print('V found - e^{2Phi}[l(l+1)/r^2 - 6m/r^3 + 4pi(rho - p_r)] =', sp.simplify(Vfound - Vclaim))
import pickle
with open('axial_potential.pkl', 'wb') as fh:
    pickle.dump(dict(V=Vfound, Vclaim=Vclaim), fh)
