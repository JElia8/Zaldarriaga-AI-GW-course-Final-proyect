"""Use a box-free 'verifbox' environment (coloured rules), avoiding minipage/tcolorbox."""
s = open('paper.tex', encoding='utf8').read()
a = s.index('\\newenvironment{verifbox}')
b = s.index('\\sloppy')
new = ('\\newenvironment{verifbox}{\\par\\medskip\\begingroup\\small\\noindent'
       '{\\color{orange!70!black}\\rule{\\linewidth}{0.8pt}}\\par\\nopagebreak\\noindent'
       '\\textbf{\\color{orange!70!black}Second-round verification.}\\ }%\n'
       '  {\\par\\nopagebreak\\noindent{\\color{orange!70!black}\\rule{\\linewidth}{0.8pt}}\\par\\endgroup\\medskip}\n')
s = s[:a] + new + s[b:]
open('paper.tex', 'w', encoding='utf8').write(s)
print('ok')
