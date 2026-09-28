# Second round: verification of the first-round work

The first round (folder `../First round`) was **not modified**. Its SHA-256 hashes were taken before any
work (`first_round_sha256_before.txt`) and after it (`first_round_sha256_after.txt`), and the two lists are identical.

## Deliverables

| file | content |
|---|---|
| `1_verification_report.pdf` | Deliverable 1: every check made, how it was verified, and the verdict (source: `report_verification/`) |
| `2_paper.pdf` | Deliverable 2: paper-style account of the whole first-round work, with all its references plus additional ones, and boxed second-round remarks (source: `paper/`) |
| `3_presentation.html` | Deliverable 3: self-contained HTML presentation (~15 min) with live simulations of the domain-wall collapse vs the static fluid shell, and a wave packet scattering off the shell. Keys: ←/→ navigate, N speaker notes, O slide list. Equations use KaTeX from a CDN (needs internet); everything else is embedded. Source: `presentation/` |
| `prompt_check.txt` | the exact prompt of this second round |

## Main findings (details in the report)

* The first round is reproducible bit for bit: `validation.json`, tables, figures and the symbolic junction matrices.
* **Confirmed independently.**
  * The background and the choice of model.
  * The odd-parity junction condition: a δ-barrier with Δ = √f(1−√f)/R. It was checked with a new thick-shell derivation.
  * The even-parity junction of the static fluid shell, i.e. model C: det T = f^{-1/2}, flux conservation, and the PSP26 matter modes. Their Newtonian limit was derived from scratch.
  * The R(ω), T(ω) and QNMs, recomputed with an independent solver.
  * The Schwarzschild checks and the WKB results.
* **Incorrect.**
  * The even-parity results of the frozen domain wall (model A) depend at O(1) on which Israel equations are imposed, so they are not an O(ω_dyn/ω) approximation.
  * The wall collapses on ~R/√2, not on (R³/M)^{1/2}.
  * The author "U.-L. Pen" should be Z. Pan (Yang, Bonga & Pan, PRL 130, 011402 (2023)).
* **Overstated or incomplete.**
  * The claimed method agreement on the QNMs: the worst difference is 6.4e-5, not ≲1e-8.
  * The QNM catalogue misses the stable matter pair at R ≤ 3M and the ℓ = 4 unstable mode at R = 2.2M.
  * The WKB errors quoted in the derivation are relative errors, which the text does not say.
* **New finding.** There is a misprint in PSP26 Eq. (7.6). Their Eq. (7.8), which the first round used, is correct.

## Folder map

* `checks/`: independent verification codes `c01`–`c09`, `ind_solver.py`, `make_round2_figures.py` and `integrity_check.sh`. Their outputs are the `out_*.txt/json` files.
* `figures_round2/`: the figures made in this round.
* `rerun/`, `rerun_validate/`, `rerun_evenC/`: copies of the first-round code, used to re-run it without touching the originals.
* `fr_code_pristine/`: an unmodified copy of the first-round code, imported read-only by the independent checks.
* `refs_text/`, `refs_img/`, `refs_pdf/`: text and page renders of the references, plus the arXiv PDFs of Pitre–Schneider–Poisson (2026) and Pani et al. (2009) that were used for the checks.
* `tmp_view/`: preview renders used while laying out the documents. They can be deleted.

## Reproducing the checks

```
cd checks
python c01_background.py            # background, I&S equations, Table 1
python c02_blackhole.py             # Leaver, BH reflectivity, Iyer-Will WKB
python c03_axial_thick_shell_derive.py && python c03b_axial_thick_shell_numeric.py
python c04_compare_scattering_qnm.py
python c05_newtonian_shell.py
python c06_even_matrix_properties.py && python c06b_followup.py && python c06c_followup.py
python c06d_modelA.py && python c06e_modelA_qnm.py
python c07_gamma_dependence.py && python c08_limits.py && python c09_poles_and_missing_modes.py
python make_round2_figures.py
sh integrity_check.sh
```
