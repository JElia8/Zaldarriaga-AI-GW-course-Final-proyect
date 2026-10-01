"""LaTeX table of the l = 0, 1 frequencies and marginal indices (writes ../report/table_modes.tex)."""
import numpy as np
from make_figures import w2_l0, w2_l1, G1, Gdip, Gcaus, catalogue_unstable

BS = '\\'
Rl2, g2 = catalogue_unstable(2)
growing = dict(zip(Rl2, g2))


def fmt(w2, RM):
    """omega M in exterior time from omega^2 R^3/M; imaginary (growing) if omega^2 < 0."""
    s = np.sqrt(abs(w2))*RM**-1.5
    return ('$%.4f$' % s) if w2 > 0 else ('$%.4f' % s + BS + ',i$')


rows = []
for RM in (2.2, 2.5, 3.0, 4.0, 6.0, 10.0, 100.0):
    x = np.sqrt(1 - 2/RM)
    l2 = growing.get(RM)
    l2s = ('$%.4f' % l2 + BS + ',i$') if l2 is not None else '--'
    rows.append('%g & %.3f & %.3f & %.2f & $%+.4f$ & %s & $%+.4f$ & %s & %s ' % (
        RM, G1(x), Gdip(x), Gcaus(x), w2_l0(x, 2.0), fmt(w2_l0(x, 2.0), RM), w2_l1(x, 2.0),
        fmt(w2_l1(x, 2.0), RM), l2s) + BS*2)
head = (BS + 'begin{tabular}{rccc|cc|cc|c}' + BS + 'toprule\n'
        ' & ' + BS + 'multicolumn{3}{c|}{marginal / causal $' + BS + 'Gamma$} & ' + BS
        + 'multicolumn{2}{c|}{$' + BS + 'ell=0$, $' + BS + 'Gamma=2$} & ' + BS
        + 'multicolumn{2}{c|}{$' + BS + 'ell=1$, $' + BS + 'Gamma=2$} & $' + BS + 'ell=2$, $' + BS + 'Gamma=2$'
        + BS*2 + '\n'
        '$R/M$ & $' + BS + 'Gamma_1$ & $' + BS + 'Gamma_{' + BS + 'rm dip}$ & $' + BS + 'Gamma_{' + BS
        + 'rm caus}$ & $' + BS + 'omega^2R^3/M$ & $' + BS + 'omega M$ & $' + BS + 'omega^2R^3/M$ & $' + BS
        + 'omega M$ & $' + BS + 'omega M$ (growing)' + BS*2 + BS + 'midrule\n')
tex = head + '\n'.join(rows) + '\n' + BS + 'bottomrule' + BS + 'end{tabular}\n'
open('../report/table_modes.tex', 'w').write(tex)
print(tex)
