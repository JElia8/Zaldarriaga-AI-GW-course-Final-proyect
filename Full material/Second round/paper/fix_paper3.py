s = open('paper.tex', encoding='utf8').read()
anchor = '\\begin{figure}[p]\\centering\n\\includegraphics[width=\\textwidth]{fig05_transfer_vs_shooting.pdf}'
add = ('\\begin{figure}[t]\\centering\n\\includegraphics[width=0.95\\textwidth]{fig12_wkb.pdf}\n'
       '\\caption{WKB validation (first round). Left: relative error of WKB orders 1, 3, 6 vs Leaver. Right: trapped modes '
       'of ultracompact shells, exact (o) vs Bohr--Sommerfeld/Gamow WKB (x).}\\label{fig:wkb}\n\\end{figure}\n\n'
       '\\begin{figure}[t]\\centering\n\\includegraphics[width=0.95\\textwidth]{fig11_modelA_adiabatic.pdf}\n'
       '\\caption{Model A with the first-round prescription. Left: relative violation of the $\\tau\\tau$ and $\\tau A$ '
       'Israel equations, $\\propto\\omega^{-1}$. Right: even-parity reflectivity of models A and C (but see '
       'Fig.~\\ref{fig:modelA2}).}\\label{fig:modelA1}\n\\end{figure}\n\n')
assert anchor in s
s = s.replace(anchor, add + anchor, 1)
s = s.replace('\\input{tables/validation.tex}', '\\resizebox{\\textwidth}{!}{\\input{tables/validation.tex}}', 1)
open('paper.tex', 'w', encoding='utf8').write(s)
print('ok')
