s = open('verification_report.tex', encoding='utf8').read()
a = 'Regge--Wheeler 1957 & Eq.~(25) (odd potential; the radial eq.\\ is (24)) & local PDF; \\OK\\\\'
assert a in s, 'row not found'
s = s.replace(a, 'Regge--Wheeler 1957 & Eq.~(25) ($k_{\\rm eff}^2$, i.e.\\ the odd potential), (24), gauge (18), (20) & '
                 'local PDF (page read); \\OK\\\\')
b = '$\\RR/[\\Delta^2/(4\\omega^2+\\Delta^2)]=0.998$--$0.999$ at $\\omega M=8$'
if b in s:
    s = s.replace(b, '$\\RR/[\\Delta^2/(4\\omega^2+\\Delta^2)]=0.997$--$0.999$ at $\\omega M=8$')
else:
    s = s.replace('=0.998$--$0.999$ at $\\omega M=8$', '=0.997$--$0.999$ at $\\omega M=8$')
open('verification_report.tex', 'w', encoding='utf8').write(s)
print('ok', s.count('0.997$--$0.999'))
