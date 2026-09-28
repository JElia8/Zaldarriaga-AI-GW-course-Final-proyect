"""CHECK 06e -- model-A QNMs under different (metric-continuity-preserving) choices of the two
imposed Israel rows, using the first round's own QNM solver (Method 1) with the row set patched."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fr_code_pristine'))
from shellgw import junction as J
from shellgw.qnm import find_qnm
import json
cat = json.load(open('../../First round/results/qnm_catalogue.json'))
c = [c for c in cat if c['parity'] == 'even' and c['model'] == 'A' and c['l'] == 2 and c['R'] == 6.0][0]
waves = sorted([m for m in c['modes'] if m['kind'] == 'wave'], key=lambda m: m['w1'][0])[:4]
oddc = [c for c in cat if c['parity'] == 'odd' and c['l'] == 2 and c['R'] == 6.0][0]
evc = [c for c in cat if c['parity'] == 'even' and c['model'] == 'C' and c['l'] == 2 and c['R'] == 6.0][0]
print('reference odd modes :', [np.round(complex(*m['w1']), 5) for m in sorted(oddc['modes'], key=lambda m: m['w1'][0])[:4]])
print('reference even-C    :', [np.round(complex(*m['w1']), 5) for m in sorted([m for m in evc['modes'] if m['kind']=='wave'], key=lambda m: m['w1'][0])[:4]])
names = {4: 'tt', 5: 'tA', 6: 'trace', 7: 'tf'}
for pair in ((6, 7), (4, 7), (5, 7), (4, 6), (4, 5)):
    J.EVOLUTION_ROWS = [0, 1, 2, 3] + list(pair)
    res = []
    for m in waves:
        g = complex(*m['w1'])
        w, r, ok = find_qnm(g, 2, 'even', 6.0, 1.0, 'A', 0.0, 1)
        res.append('%.5f%+.5fi%s' % (w.real, w.imag, '' if (ok and r < 1e-8) else '(?)'))
    print('rows %-9s:' % (names[pair[0]] + '+' + names[pair[1]]), ', '.join(res))
