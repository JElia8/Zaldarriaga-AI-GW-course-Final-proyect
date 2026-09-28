"""Publication-style plotting helpers (PDF + PNG output)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')

# colour-blind-safe palette (Okabe-Ito)
C = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000', '#F0E442']

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 8.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.grid': False,
    'lines.linewidth': 1.4,
    'savefig.bbox': 'tight',
    'figure.dpi': 110,
    'mathtext.fontset': 'cm',
})


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGDIR, '%s.%s' % (name, ext)), dpi=250 if ext == 'png' else None)
    plt.close(fig)
    return os.path.join(FIGDIR, name + '.pdf')
