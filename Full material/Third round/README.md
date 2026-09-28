# Third round: summary, paper and presentation

| file | content |
|---|---|
| `1_summary.pdf` | 5-page summary for the course: main ideas, line of thought, figures and results (source `summary/`) |
| `2_paper.pdf` | Full paper. It is written as eight explicit steps, each a *Question*, a *Key idea* and a *What we learned* box (source `paper/`) |
| `3_presentation.html` | Self-contained HTML talk (~15 min, 18 slides) with the interactive collapse animation and live wave-packet simulation. Keys: ←/→, N notes, O overview. Equations use KaTeX from a CDN (needs internet). Source: `presentation/` |
| `prompt_third_round.txt` | the exact prompt of this round |

## Folders
* `figures/`: new single-panel figures (PDF + PNG), all made by `code/make_figures_r3.py`.
* `code/make_figures_r3.py`: reads the first-round results (read only) and a copy of the first-round code (`code/fr_code/`). It also computes:
  * R(ω) up to ωM = 12, to test the high-frequency δ-barrier law. Agreement is better than 0.5 %.
  * a time-domain (leapfrog) simulation of a pulse on a black hole and on shells at R = 3M and 2.2M.
    * Black-hole ringdown fit: 0.3731 − 0.0885i, against Leaver's 0.3737 − 0.0890i.
    * R = 2.2M trapped-mode fit: 0.4215 − 0.0352i, against the frequency-domain 0.4211 − 0.0350i.
  * the Newtonian matter-mode roots, 1.5050 and 0.5147i.

  The quoted numbers are saved in `code/numbers_r3.json`.
* `presentation/build_presentation.py`: assembles the HTML from `slides_new.html` and the style/simulation code of the
  second-round template (read only), and embeds the figures.

## Rebuild
```
cd code && python -B make_figures_r3.py
cd ../summary && pdflatex summary.tex && pdflatex summary.tex
cd ../paper && pdflatex paper.tex && pdflatex paper.tex && pdflatex paper.tex
cd ../presentation && python -B build_presentation.py
```
Nothing in `First round/`, `Second round/`, `Starting point/` or `Bibliography/` was modified.
