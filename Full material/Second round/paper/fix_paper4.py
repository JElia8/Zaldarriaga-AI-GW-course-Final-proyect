s = open('paper.tex', encoding='utf8').read()
a = s.index('\\text{outside:}&\\quad \\delta h_{\\tau\\tau}=H_+')
b = s.index('\\end{align}', a)
new = ('\\text{outside:}&\\quad \\delta h_{\\tau\\tau}=H_+-\\tfrac{2M}{R(R-2M)}\\xi_+,\\qquad \\delta h_{\\tau A}=0,\\nonumber\\\\\n'
       '&\\quad \\delta h^{\\rm tr}_{AB}=R^2K_++2R\\xi_+,\\qquad \\delta h^{\\rm tf}_{AB}=0,\\nonumber\\\\\n'
       '\\text{inside:}&\\quad \\delta h_{\\tau\\tau}=H_-+2i\\nu\\zeta,\\qquad \\delta h_{\\tau A}=-\\zeta-i\\nu R^2\\eta,\\nonumber\\\\\n'
       '&\\quad \\delta h^{\\rm tr}_{AB}=R^2K_--\\lambda R^2\\eta+2R\\xi_-,\\qquad \\delta h^{\\rm tf}_{AB}=2R^2\\eta .\n')
s = s[:a] + new + s[b:]
open('paper.tex', 'w', encoding='utf8').write(s)
print('ok')
