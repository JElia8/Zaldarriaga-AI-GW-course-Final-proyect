# Final round

| file | content |
|---|---|
| `1_polarization_parity_report.pdf` | Report (15 pp.) on h_+, h_x and the two parities: the dictionary between polarizations and parities, the selection rule, the "birefringent mirror" (black hole and shells), the helicity-flip cross section, reflected waveforms in the two parities, and which parity matters (source: `report/`) |
| `2_summary.pdf` | Self-contained 5-page summary of the whole project (source: `summary/`). **Final**: no new results will be added. |
| `3_presentation.html` | Self-contained talk, 26 slides (source: `presentation/`). It is the only document that may still be revised. |
| `prompt_fourth_round.txt`, `prompt_final_step.txt` | the prompts of this round |

## Presentation
* Keys: arrow keys to move, N for speaker notes, O for the slide overview. Equations use KaTeX from a CDN (internet needed); everything else is embedded.
* Interactive slides:
  * collapse of a domain wall vs the static fluid shell;
  * a live wave packet scattering off the shell;
  * the polarization explorer: retardance and reflected polarization vs frequency for a black hole and three shells.
* To edit: change `presentation/deck_source.html`, then run `python presentation/build_presentation.py`. The script embeds the figures from `figures/` and the data from `presentation/pol_data.json`, and writes both `presentation/presentation.html` and `3_presentation.html`.

## Code
* `code/parity_study.py`: all polarization computations (cached in `data/`).
* `code/residue.py`: residue of S_even at the unstable pole.
* `code/make_figures_r4.py`: figures.

The physics modules are imported read-only from `../Third round/code/fr_code`, and the scattering data from `../First round/results`, so this folder must stay next to the other rounds inside `Full material/`.
