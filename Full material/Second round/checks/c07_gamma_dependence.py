"""CHECK 07 -- Gamma dependence of the l=2 matter modes at M/R = 0.2 (compare PSP26 Figs. 1-2, right/lower panels),
independent solver with the first-round even junction."""
import ind_solver as I
R = 5.0; sc = R**-1.5
print('PSP26 Fig.1/2 (M/R=0.2, l=2) Gamma dependence, my solver (first-round T_even), w (R^3/M)^1/2:')
wu = 0.6077j*sc; ws = (1.267 - 0.0011j)*sc
for G in (1.81, 1.9, 2.0, 2.1, 2.2, 2.3, 2.46):
    wu = I.qnm(wu, 2, 'even', R, 'C', Gamma=G); ws = I.qnm(ws, 2, 'even', R, 'C', Gamma=G)
    print('  Gamma=%.2f  unstable %.4fi   stable %.4f %+.2ei' % (G, (wu/sc).imag, (ws/sc).real, (ws/sc).imag))
