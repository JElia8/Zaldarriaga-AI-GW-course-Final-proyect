# Gravitational-wave perturbations of a spherical thin shell

Flat interior, Schwarzschild exterior. The project computes the scattering coefficients R(ω), T(ω) and
the quasinormal modes, for odd (Regge–Wheeler / CPM) and even (Zerilli–Moncrief) parity.

* **Derivation document:** `derivation/derivation.pdf` (source `derivation/derivation.tex`). It is
  self-contained, with numbered equations and literature citations.
* **Code:** `code/` (Python 3.12; numpy, scipy, sympy, mpmath, matplotlib).
* **Figures:** `figures/` (PDF + PNG).
* **Results:** `results/` (JSON, `qnm_table.csv`, logs of every symbolic verification).

## Shell models (read this first)

| model | shell | background | regime of validity |
|---|---|---|---|
| **C** (primary) | static perfect fluid, σ = (1−√f)/(4πR), p = κσ, κ = (1−√f)/(4√f) (Poisson 2004 Eqs. 3.80–3.81) | exactly static, but **linearly unstable** (even-parity matter mode, Im ω ≈ 0.6 (M/R³)^½; Pitre, Schneider & Poisson 2026, reproduced here) | ω ≫ (M/R³)^½; R ≥ 2.1M (energy and causality conditions) |
| **A** | pure domain wall τ = σ (Ipser & Sikivie 1984) | frozen at its turning point, R̈ = −(1+3√f)/(2R) | ω ≫ ω_dyn; error O(ω_dyn/ω), quantified by the code |
| B (dust) | — | **does not exist**: a static dust shell is impossible (I&S Eq. 3.7a) | not implemented |

The odd-parity junction conditions are the same for every model:
Ψ₊ = Ψ₋ and √f Ψ₊′ − Ψ₋′ = (1−√f)Ψ/R. This is a repulsive δ-function of strength
Δ = √f(1−√f)/R in the tortoise coordinate.
The even-parity junction conditions form a frequency-dependent 2×2 transfer matrix, derived symbolically
from first principles.

## Running

```
cd code
python symbolic/vacuum_relations.py        # (run inside code/symbolic) metric <-> master-function maps, verified
python symbolic/exterior_series.py         # continued-fraction recurrence, Chandrasekhar map, verified
python symbolic/derive_junction.py         # junction conditions (writes shellgw/junction_*.py)
python main.py validate scatter qnm tracks csv figures
python make_tables.py                      # LaTeX tables for the derivation document
cd ../derivation && pdflatex derivation.tex && pdflatex derivation.tex
```

## Validation (all in `results/validation.json`, summarized in the derivation, Sec. 6.1)

* **Energy conservation:** |R+T−1| ≲ 10⁻¹⁰ (odd, even model C). For model A the violation decreases as
  ω⁻² (an adiabatic error, not a numerical one).
* **Method agreement:** transfer matrix vs shooting agree to ≲10⁻¹⁰ on R and T, and to ≲10⁻⁸ on QNMs.
* **Schwarzschild QNMs:** l = 2, n = 0 gives 0.37367168 − 0.08896232i (Leaver continued fraction). All
  n ≤ 3 modes for l = 2, 3, 4 match Berti et al. (2009).
* **Black-hole reflectivity:** reproduces Vishveshwara (1970). The shell's R(ω) tends to the black-hole
  value as R → 2M.
* **WKB:** orders 1, 3 and 6 reproduce Iyer & Will (1987) and Konoplya (2003).
* **Literature cross-check:** the even-parity matter modes and the wave modes reproduce Pitre, Schneider
  & Poisson (2026, arXiv:2604.05980), who studied the same static shell.
