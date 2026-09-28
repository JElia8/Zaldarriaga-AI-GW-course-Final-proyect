"""Post-processing of results/qnm_catalogue.json:
 * de-duplicate modes (|dw| < 1e-7),
 * re-converge with Method 1 (40-digit junction for R < 2.3M) and Method 2,
 * record the residuals and the M1-M2 difference."""
import numpy as np
from multiprocessing import Pool
from shellgw import studies as S
from shellgw.qnm import find_qnm


def polish(c):
    vs2 = c['vs2']
    out = []
    for m in c['modes']:
        w0 = complex(*m['w1'])
        w1, r1, ok1 = find_qnm(w0, c['l'], c['parity'], c['R'], 1.0, c['model'], vs2, method=1)
        if not (ok1 and r1 < 1e-8):
            w1, r1 = w0, m['res']
        if any(abs(w1 - complex(*o['w1'])) < 1e-7 for o in out):
            continue
        w2, r2, ok2 = find_qnm(w1, c['l'], c['parity'], c['R'], 1.0, c['model'], vs2, method=2)
        out.append(dict(kind=m['kind'], n=m['n'], w1=S.cplx(w1), w2=S.cplx(w2), diff=abs(w1 - w2), res=r1,
                        shift_from_scan=abs(w1 - w0)))
    wave = sorted([m for m in out if m['kind'] == 'wave'], key=lambda m: m['w1'][0])
    for n, m in enumerate(wave):
        m['n'] = n
    c = dict(c)
    c['modes'] = out
    return c


if __name__ == '__main__':
    cat = S._load('qnm_catalogue.json')
    with Pool(10) as p:
        new = p.map(polish, cat, chunksize=1)
    S._save('qnm_catalogue.json', new)
    tot = sum(len(c['modes']) for c in new)
    bad = [(c['parity'], c['model'], c['l'], c['R'], m['w1'], m['diff']) for c in new for m in c['modes'] if m['diff'] > 1e-4]
    print('modes', tot, 'M1-M2 > 1e-4:', len(bad))
    for b in bad:
        print('  ', b)
    print('max shift from scan:', max(m['shift_from_scan'] for c in new for m in c['modes']))
