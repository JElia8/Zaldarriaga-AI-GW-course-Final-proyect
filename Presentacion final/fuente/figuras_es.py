"""Figura de la reflectividad (impar, l = 2) del cascarón frente al agujero negro, en castellano y con el formato
de la diapositiva (sin título, letras grandes, márgenes mínimos). Lee, sin modificarlos, los datos de
../../Full material/First round/results/scattering.json y escribe ../figuras/r_limite_bh_es.png.
Correr:  python -B figuras_es.py"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'Full material', 'First round', 'results', 'scattering.json')
OUT = os.path.join(HERE, '..', 'figuras', 'r_limite_bh_es.png')
VIOLET, BLUE, AQUA, YELLOW = '#4a3aa7', '#2a78d6', '#1baf7a', '#eda100'
plt.rcParams.update({'font.family': 'serif', 'font.size': 15, 'axes.labelsize': 17, 'legend.fontsize': 13.5,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'mathtext.fontset': 'cm', 'legend.frameon': False})

scat = json.load(open(DATA))
bh = next(b for b in scat['bh'] if b['l'] == 2 and b['parity'] == 'odd')


def curve(R):
    return next(c for c in scat['curves'] if c['l'] == 2 and abs(c['R'] - R) < 1e-9 and c['parity'] == 'odd'
                and c['model'] == 'C' and abs(c['Gamma'] - 2.0) < 1e-9)


fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(bh['w'], bh['R'], color='k', lw=5, alpha=0.28, label='agujero negro')
for R, col in zip((3.0, 2.2, 2.05, 2.01), (YELLOW, AQUA, BLUE, VIOLET)):
    c = curve(R)
    ax.plot(c['w'], c['R1'], color=col, lw=2.2, label=r'cascarón $R=%gM$' % R)
ax.set_xlim(0.15, 0.75)
ax.set_ylim(-0.02, 1.03)
ax.set_xlabel(r'$\omega M$')
ax.set_ylabel(r'reflectividad $\mathcal{R}(\omega)$')
ax.legend(loc='upper right')
fig.savefig(OUT, dpi=200, bbox_inches='tight', pad_inches=0.04)
print('escrita', OUT)


# ---------------------------------------------------------------- modos cuasinormales: agujero negro vs cascarones
# Agujero negro, l = 2, n = 0..3 (Leaver 1985). Cascarones R = 3M y 2.2M: catálogo de modos impares
# (results/qnm_catalogue.json). R = 2.01M: los tres modos atrapados de derivation/tables/trapped.tex (exactos, M1).
cat = json.load(open(os.path.join(HERE, '..', '..', 'Full material', 'First round', 'results', 'qnm_catalogue.json')))
BH = [(0.37367, -0.08896), (0.34671, -0.27391), (0.30105, -0.47828), (0.25150, -0.70514)]
TRAP201 = [(0.162603, -9.711e-07), (0.248683, -5.284e-05), (0.326133, -8.798e-04)]


def shell_modes(R):
    c = next(c for c in cat if c['l'] == 2 and abs(c['R'] - R) < 1e-9 and c['parity'] == 'odd' and c['model'] == 'C')
    return [tuple(m['w1']) for m in c['modes']]


fig, ax = plt.subplots(figsize=(6.4, 4.0))
for data, col, mk, lab, ms in [(shell_modes(3.0), YELLOW, 'o', r'cascarón $R=3M$', 9),
                               (shell_modes(2.2), AQUA, 'o', r'cascarón $R=2.2M$', 9),
                               (TRAP201, VIOLET, 'o', r'cascarón $R=2.01M$', 9),
                               (BH, 'k', '*', 'agujero negro', 16)]:
    x = [w[0] for w in data]; y = [1 / abs(w[1]) for w in data]
    ax.scatter(x, y, s=ms**2, marker=mk, facecolor=col if mk == '*' else 'none', edgecolor=col, lw=2, label=lab, zorder=3)
ax.set_yscale('log')
ax.set_xlim(0.1, 1.25)
ax.set_ylim(0.6, 3e6)
ax.set_xlabel(r'frecuencia  ${\rm Re}\,\omega M$')
ax.set_ylabel(r'vida  $1/|{\rm Im}\,\omega|$  $(M)$')
ax.legend(loc='upper right', fontsize=12.5)
fig.savefig(os.path.join(HERE, '..', 'figuras', 'modos_es.png'), dpi=200, bbox_inches='tight', pad_inches=0.04)
print('escrita modos_es.png')

# ---------------------------------------------------------------- reflectividad par vs impar (interior abierto)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(bh['w'], bh['R'], color='k', lw=5, alpha=0.28, label='agujero negro (par = impar)')
for R, col in ((6.0, BLUE), (3.0, YELLOW)):
    for par, ls, nm in (('even', '-', 'par'), ('odd', '--', 'impar')):
        c = next(c for c in scat['curves'] if c['l'] == 2 and abs(c['R'] - R) < 1e-9 and c['parity'] == par
                 and c['model'] == 'C' and abs(c['Gamma'] - 2.0) < 1e-9)
        ax.plot(c['w'], c['R1'], color=col, ls=ls, lw=2.2, label=r'$R=%gM$, %s' % (R, nm))
ax.set_yscale('log')
ax.set_xlim(0.05, 1.0)
ax.set_ylim(2e-7, 2)
ax.set_xlabel(r'$\omega M$')
ax.set_ylabel(r'reflectividad $\mathcal{R}(\omega)$')
ax.legend(loc='lower left', fontsize=11.5, ncol=2, columnspacing=1.0)
fig.savefig(os.path.join(HERE, '..', 'figuras', 'par_impar_es.png'), dpi=200, bbox_inches='tight', pad_inches=0.04)
print('escrita par_impar_es.png')
