# Radial (ℓ = 0) and dipole (ℓ = 1) oscillations of the static thin shell

Question: does the static perfect-fluid shell (flat interior, Schwarzschild exterior, p = σ(1−√f)/4√f) oscillate
or collapse under its lowest-multipole perturbations? These modes do not radiate, so ω² is real. This is the first
step toward the collapsing-shell problem (the thin-shell version of gravitational collapse in Landau & Lifshitz,
Vol. 2).

| file | content |
|---|---|
| `report.pdf` | 5-page report: derivations, results, figures, validation (source `report/report.tex`) |
| `prompt.txt` | the prompt of this step |

## Main results (x = √f, Γ = adiabatic index, ω in exterior time)
* **ℓ = 0:** ω₀²R³/M = 2f/(1+x) · (Γ − Γ₁), with Γ₁ = (1 + 2x + 3f)/(4f). The marginal index equals the turning-point
  criterion of Pitre, Schneider & Poisson (2026, Eq. 3.16). The Newtonian limit is Γ − 3/2.
* **ℓ = 1, even:** ω₁²R³/M = 6f/(1+3x) · (Γ − Γ_dip), with Γ_dip = Γ₁ − 1/(6f). There is also a double zero mode
  (translation and boost). The Newtonian limit is (3Γ − 4)/2, as in PSP26 Eq. 10.57.
* **ℓ = 1, odd:** no oscillation (no odd matter mode at any ℓ). The only solution is a slow rigid rotation, which drags
  the interior at Ω_drag = 2J/R³, with Ω_drag/Ω_shell = (1−x)(1+3x)/(1+2x). This tends to 4M/3R (Thirring) in the weak
  field and to 1 as R → 2M.
* **For Γ = 2** (the value used throughout the project): radially unstable for R < 3.8165M, dipole unstable for
  R < 2.767M. For R < 2.370M no causal equation of state gives radial stability.
* **Nonlinear:** a radially unstable shell pushed inward collapses through r = 2M (black-hole formation). Pushed
  outward, it makes a large bounded excursion. A stable shell oscillates, and its measured frequency agrees with the
  linear one to 0.03%.

## Code (`code/`, Python 3.12 + sympy/scipy/matplotlib)
```
cd code
python -B l0_potential.py      # exact l=0 potential -> omega_0^2, Gamma_1, Newtonian limit
python -B israel_low_l.py      # linearized Israel equations (unperturbed metrics, perturbed embeddings), l=0,1
python -B modes_low_l.py       # determinants -> omega_0^2 (check) and omega_1^2, zero mode, eigenvector
python -B check_gauge.py       # same frequencies with the gauge fixed on the other side
python -B check_newtonian.py   # Newtonian shell equations at l=0,1
python -B l1_odd_rotation.py   # odd l=1: rotation and frame dragging
python -B l0_nonlinear.py      # nonlinear radial evolution (polytrope), about 2 min
python -B make_figures.py      # figures/ (reads the first-round QNM catalogue for the l>=2 rates)
python -B make_table.py        # report/table_modes.tex
cd ../report && pdflatex report.tex && pdflatex report.tex
```
Logs and numbers are in `results/`. `make_figures.py` reads (read only)
`../Full material/First round/results/qnm_catalogue.json` for the ℓ = 2, 3 growth rates. Nothing outside this folder
was modified.
