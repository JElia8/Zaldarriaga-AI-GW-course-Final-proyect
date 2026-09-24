# Gravitational Wave Perturbations of a Spherical Thin Shell
## Prompt for Claude Code

---

## Folder structure

```
project/
├── PROMPT.md                  ← this file
├── Bibliography/              ← primary reference papers (read all of them)
│   ├── Regge_Wheeler_1957
│   ├── Zerilli_1970
│   ├── Vishveshwara_1970
│   ├── Martel_Poisson_2005
│   ├── Martel_thesis
│   ├── Ipser_Sikivie_1983
│   ├── Poisson_toolkit
│   └── Boyanov_Cardoso_2024   ← closely related scattering calculation (read carefully)
└── Starting_point/            ← prior work done on this problem (read carefully)
    ├── interior_ondas_cascara  ← interior wave solutions derived
    └── burbuja_matching_israel ← background dynamics and junction condition setup
```

Read **everything** in both folders before writing any code. The `Starting_point/`
documents are not necessarily the path you must follow — they are prior work that
documents what has already been established and, critically, what problems were
encountered. You may build on them, correct them, or take a different approach,
but you must engage with the physics they raise.

---

## The research problem

A static spherical thin shell at areal radius `r = R` separates:

- **Interior** (`r < R`): flat Minkowski spacetime, `ds² = −dt² + dr² + r²dΩ²`
- **Exterior** (`r > R`): Schwarzschild spacetime of mass `M` (`R > 2M`),  
  `ds² = −f(r)dt² + f(r)⁻¹dr² + r²dΩ²`, `f(r) = 1 − 2M/r`

The goal is to compute:
1. **Reflection and transmission coefficients** `R(ω)` and `T(ω)` for
   gravitational waves scattering off the shell.
2. The **quasi-normal mode (QNM) spectrum** of the system — complex
   frequencies at which outgoing-only solutions exist.

Both quantities must be computed for **odd-parity (Regge-Wheeler)** and
**even-parity (Zerilli)** perturbations, and for multiple values of `l` and
shell parameters.

All results must be accompanied by publication-quality plots and a
self-contained written derivation document.

Work in geometric units `G = c = 1`. Express frequencies as `ωM`.

---

## What is already established — read the Starting_point/ documents

The `Starting_point/` folder contains two documents produced in prior work:

**`interior_ondas_cascara`** establishes:
- The interior is exactly Minkowski (Birkhoff + regularity at `r = 0` forces
  `M_int = 0`).
- Both the even-parity (Zerilli-Moncrief) and odd-parity (Cunningham-Price-
  Moncrief) master functions obey the **same** wave equation inside the shell:
  `Ψ'' + [ω² − l(l+1)/r²] Ψ = 0` (prime = d/dr, since r* = r in flat space).
- The regular solution at `r = 0` is `Ψ_interior = A · ĵ_l(ωr)`, where
  `ĵ_l(x) = x j_l(x)` is the Riccati-Bessel function, regular as `r^(l+1)`.
- The two parities share the same radial profile but carry independent complex
  amplitudes and will be distinguished by the junction conditions at `r = R`.

**`burbuja_matching_israel`** establishes:
- The general Gauss-Codazzi-Israel thin-shell formalism, following Ipser &
  Sikivie (1984).
- For a domain wall, `S^ab = −σ h^ab` with `σ = const`. The surface
  stress tensor has no independent dynamical degrees of freedom beyond the
  embedding.
- **The collapse problem**: for a pure domain wall (tension `τ = σ`) with
  nonzero `M`, the background equation of motion gives `R̈ < 0` always.
  There is **no static equilibrium**. The shell necessarily collapses to a
  black hole.
- The adiabatic/high-frequency approximation is proposed: freeze the background
  at the momentarily-static turning point `R = R_m`, `Ṙ = 0`, and compute
  `R(ω)`, `T(ω)` treating the shell as static — valid when `ω × (shell-crossing
  time) ≫ 1`.
- Partial progress on the linearized odd-parity junction condition: the
  perturbed extrinsic curvature component `δK_tA` has been computed; the
  remaining steps (evaluating the background trace jump `[K]^(0)` at `R_m`,
  and expressing everything in terms of the master function `Ψ_odd` and its
  derivative) are identified but not yet completed.

---

## The collapse problem — you must address this explicitly

This is the central physical subtlety of the problem. **Before writing any
scattering code**, you must make a deliberate, justified decision about how to
handle the fact that a pure domain wall does not admit a static background.

Read Ipser & Sikivie (1984) Sections II and III carefully. The background
equation of motion for a spherical shell with general `(σ, τ)` is (their
Eq. 3.7a):

```
(α + β) R̈ = − αM/R² − (2τ/σ) αβ(α+β)/R
```

where `α = (1 + Ṙ²)^(1/2)` and `β = (f(R) + Ṙ²)^(1/2)`.

Setting `Ṙ = Ṙ = 0` (static shell), this requires:

```
M/R² + (2τ/σ) β²/R = 0
```

Since `M, R, β > 0`, a static solution requires `τ < 0` (negative tension) or
`M = 0`. A pure domain wall has `τ = σ > 0` — **no static solution exists**
for `M ≠ 0`.

**You must evaluate the following options and choose one (or more) to implement,
with explicit physical justification:**

### Option A — Adiabatic approximation (domain wall, momentarily static)

Freeze the shell at its turning point `R_m`, `Ṙ = 0` (but `R̈ ≠ 0`). Use the
mass formula (Ipser & Sikivie Eq. 3.9):

```
M = 4πσ R_m² (1 − 2πσ R_m)
```

to relate `M`, `σ`, `R_m`. Compute the scattering problem with the shell
treated as static. This is valid at high frequencies `ω ≫ |R̈/Ṙ|` near the
turning point (formally `ω → ∞`, but practically once `ω` exceeds the shell's
dynamical frequency scale).

**Important**: at the turning point `Ṙ = 0` but `R̈ ≠ 0`, so the background
trace jump `[K]^(0)` is nonzero (it includes an `R̈`-dependent term). This
must be correctly included in the linearized junction conditions — the
`burbuja_matching_israel` document identifies this as the remaining step.

### Option B — Dust shell (static by construction)

A pressureless dust shell (`τ = 0`) can be in genuine static equilibrium.
Setting `τ = 0` and `Ṙ = Ṙ = 0` in the equations of motion gives:

```
M = 4πσ R² √f(R)  →  σ = M / (4πR² √f(R))
```

This is a true static background with `[K]^(0)` well-defined and simpler
(no `R̈` contribution). The junction conditions for perturbations are
conceptually cleaner. This is a physically distinct shell (pressureless dust,
not a domain wall) but it is a legitimate and well-studied system.

### Option C — General (σ, τ) family, static shell

Search for values of `τ/σ` that admit static equilibrium with `M ≠ 0`. From
the equations of motion with `Ṙ = Ṙ = 0`: this requires `τ/σ < 0` in
general. Parameterize by `τ/σ = −κ` with `κ > 0` and study the scattering
as a function of `κ`. This gives a family of shell types that continuously
interpolates through physically realizable configurations.

### Option D — Schwarzschild star interior (gravastar-like)

Replace the Minkowski interior with a Schwarzschild interior of mass `M_in < M`,
allowing a genuine static shell via the pressure of the interior. This is a more
physical model but requires revisiting the interior solution (no longer flat,
Bessel functions are replaced by confluent Heun functions or similar). This is
the most complex option.

**You are not required to implement all four options.** Make a clear choice,
justify it physically, and implement it completely. Implementing Options A and B
in parallel (comparing their QNM spectra and R/T coefficients) is a natural
and informative combination.

**Whatever you choose, state explicitly at the start of the code and the
derivation document**: what shell model you are using, whether it is truly
static or adiabatically frozen, and what approximation regime the results are
valid in.

---

## Physics verification — mandatory

Every non-trivial claim in the code and the derivation document must be
verified against the literature. This is not optional. Specifically:

**Cross-check against papers in `Bibliography/`:**
- Every equation you use must be traceable to a specific equation number in
  one of the reference papers. Write this as a comment in the code
  (e.g. `# Martel & Poisson (2005) Eq. (5.15)`) and as a citation in the
  derivation document.
- The Zerilli potential: verify your implementation against Zerilli (1970)
  Eq. (5) and Martel & Poisson (2005) Eq. (4.26).
- The Regge-Wheeler potential: verify against Regge & Wheeler (1957) Eq. (25)
  and Martel & Poisson (2005) Eq. (5.15).
- The interior Bessel solutions: verify against `interior_ondas_cascara` and
  Abramowitz & Stegun §10.3.
- The background junction conditions: verify against Ipser & Sikivie (1984)
  Eqs. (2.6), (3.5)–(3.9).
- The linearized junction conditions for perturbations: this is the least
  documented part. Derive them carefully from first principles following the
  Gauss-Codazzi formalism. Verify each step symbolically if possible (SymPy
  can help).

**Independent validation checks in the code:**
- **Energy conservation**: verify `R(ω) + T(ω) = 1` for real `ω` to better
  than `10⁻⁶` across the full frequency range.
- **Pure Schwarzschild limit**: in the limit of a transparent shell (zero
  junction condition coupling, or `σ → 0`), recover the standard Schwarzschild
  reflection coefficient from Vishveshwara (1970). The `l=2, n=0` QNM must
  satisfy `ωM ≈ 0.3737 − 0.0890i` (known to many decimal places in the
  literature — look up the precise value).
- **Low-frequency limit**: for `ω → 0`, `R → 1` and `T → 0` (the shell
  becomes opaque at zero frequency for a gravitational wave).
- **WKB vs. numerical**: QNM frequencies from the WKB approximation
  (Schutz-Will 1985, Iyer-Will 1987) must agree with your numerical values
  to the expected WKB accuracy for the chosen `l`.
- **Consistency of the two numerical methods** (transfer matrix and shooting):
  they must agree on `R`, `T`, and QNM frequencies to better than `10⁻⁴`.

**Use Boyanov, Cardoso, Kokkotas & Redondo-Yuste (2024) as a methodological
guide** (`Boyanov_Cardoso_2024` in `Bibliography/`). This paper studies GW
scattering off a compact viscous star — not a thin shell, but the exterior
spacetime and numerical framework are identical to ours. Read it carefully
for the following specific inputs:

- **Numerical setup**: their frequency-domain shooting method (Appendix B1)
  and time-domain method (Appendix B2) are directly reusable. In particular,
  their reflectivity extraction formula (Eq. 17) and the technique of
  evaluating `|iω + ψ'/ψ| / |iω − ψ'/ψ|` averaged over several oscillation
  periods is a clean way to extract `R(ω)` without fixing a large but finite
  outer boundary. Adopt this.
- **QNM shooting**: their method of integrating inward from a high-order
  exterior expansion to a midpoint, then demanding continuity with the interior
  integration (Appendix B1), is more stable than naive asymptotic extraction.
  Use a similar approach for your QNM root-finding.
- **Inviscid limit as a cross-check**: in the limit `η → 0` (zero viscosity),
  their interior is a perfect fluid that transmits the wave without absorption.
  Their junction condition (Eq. 14) reduces in this limit to simple continuity
  of `ψ` and `dψ/dr*` — which must match the thin-shell limit of your junction
  conditions when the shell has zero coupling (`σ → 0`). Verify this
  explicitly.
- **Physical expectations**: their Fig. 1 shows that at low frequencies
  `R² → 1` regardless of compactness or viscosity — a general feature of
  long-wavelength scattering that your thin-shell result must also reproduce.
  At high frequencies their inviscid star is perfectly transmitting (`R² = 1`
  trivially, since T = 0 and R = 1 for a lossless object in their setup).
  For your shell, the high-frequency behaviour of R(ω) will be different —
  think carefully about what it should be.

**Do not import their interior physics**: their interior master equation
(Eqs. 10–11) is a coupled system for the GW master function and a viscous
fluid variable `β`. Your interior has no matter, so there is no `β` and the
interior equation is the simple Bessel equation already established in
`interior_ondas_cascara`. The junction condition (their Eq. 14) involves `η`
explicitly and is not yours — your junction condition is derived from the
Israel thin-shell formalism, not from a fluid surface condition.

**Search for additional references online:**
If you encounter a step where the `Bibliography/` papers are insufficient,
search for papers that cite Regge & Wheeler (1957), Zerilli (1970), or
Ipser & Sikivie (1984). In particular:
- Papers studying gravitational perturbations of thin-shell spacetimes (search
  "thin shell gravastars quasi-normal modes", "gravastar perturbations", or
  "thin shell scattering gravitational waves").
- Papers on QNMs of compact objects with Minkowski interior and Schwarzschild
  exterior — these are sometimes called "gravastars" or "black hole mimickers"
  in the literature, and there is a body of work you should compare against.
- If you find a published result for the R/T coefficients or QNMs of a
  Schwarzschild exterior / flat interior system, compare your numerical results
  against them quantitatively.

Document every external reference you use (author, title, journal, year,
equation number) in the derivation document.

---

## Perturbation equations — use these conventions

Follow Martel & Poisson (2005) throughout. Their gauge-invariant master
functions are the standard modern reference.

**Exterior (r > R), Schwarzschild:**

Odd parity — Cunningham-Price-Moncrief function `Ψ_odd`:
```
d²Ψ/dr*² + [ω² − V_RW(r)] Ψ = 0
V_RW = f(r) [l(l+1)/r² − 6M/r³]         (Martel & Poisson Eq. 5.15)
r* = r + 2M ln|r/2M − 1|
```

Even parity — Zerilli-Moncrief function `Ψ_even`:
```
d²Ψ/dr*² + [ω² − V_Z(r)] Ψ = 0
V_Z = f(r) [2n²(n+1)r³ + 6Mn²r² + 18M²nr + 18M³] / [r³(nr + 3M)²]
n = (l−1)(l+2)/2                          (Zerilli 1970 Eq. 5)
```

**Interior (r < R), Minkowski:**

Both parities obey (established in `interior_ondas_cascara`):
```
d²Ψ/dr² + [ω² − l(l+1)/r²] Ψ = 0
```
Regular solution: `Ψ = A · ĵ_l(ωr)`, `ĵ_l(x) = x j_l(x)`

**Asymptotic boundary conditions:**
- Exterior, `r* → +∞`: `Ψ → B_in e^{−iωr*} + B_out e^{+iωr*}`
- Interior, `r → 0`: `Ψ → A · ĵ_l(ωr) ~ A · (ωr)^{l+1}/(2l+1)!!`

Be careful and consistent with the sign convention `e^{−iωt}` throughout.

---

## Junction conditions — derive these carefully

The junction conditions at `r = R` take the form:
```
[Ψ]     = 0                              (continuity)
[dΨ/dr*] = Δ(ω, l, R, M) · Ψ(R)        (derivative jump)
```

where `[X] = X_ext(R⁺) − X_int(R⁻)` and `Δ` depends on the shell model.

**This derivation is the central technical challenge of the problem.** The
`burbuja_matching_israel` document has done the groundwork for odd parity:
the perturbed extrinsic curvature `δK_tA` has been computed (their Eq. 12),
and the structure of the matching equation (their Eq. 13) has been identified.
What remains is:

1. Evaluate `[K]^(0)` (the background trace jump) for your chosen shell model.
   For a dust shell at static equilibrium, `[K]^(0)` is straightforward. For
   the adiabatic domain wall at the turning point, it involves `R̈` evaluated
   from Eq. (3.7a) of Ipser & Sikivie.
2. Express `h_t^lm` and `h_r^lm` on each side in terms of the master function
   `Ψ_odd` using Martel & Poisson (2005) Eqs. (5.7) and (5.13).
3. Reduce to a junction condition purely in `Ψ_odd(R)` and `dΨ_odd/dr*(R)`.

For even parity, the derivation is analogous but uses the Zerilli-Moncrief
function and the even-parity perturbed extrinsic curvature. The even-parity
sector is generally more complex because the shell can also be radially
displaced by even-parity perturbations.

Verify the final `Δ` coefficients symbolically (SymPy) and check that they
reduce to known limits (e.g., `Δ → 0` for a transparent shell).

---

## Numerical methods — implement at least two and compare

### Method 1: Transfer matrix

Construct the solution analytically in each region:
- Interior: `Ψ_int = A · ĵ_l(ωr)`
- Exterior: `Ψ_ext = B_in · f_in(r*) + B_out · f_out(r*)` where `f_in/out`
  are the ingoing/outgoing Schwarzschild solutions (computed numerically, or
  via Leaver's continued-fraction method for QNMs).

Apply the two junction conditions at `r = R` to get two equations in the
unknowns. Solve the linear system for the scattering coefficients `B_out/B_in`
(reflection) and `A/B_in` (transmission).

### Method 2: Shooting method

Integrate the ODE from `r* → −∞` (interior, using regularity at origin as
initial condition), applying the junction-condition jump at `r = R`, then
continuing to `r* → +∞`. Extract asymptotic amplitudes by projecting onto
`e^{±iωr*}` over a window of several wavelengths.

Use `scipy.integrate.solve_ivp` with `DOP853` method, tolerances `rtol=1e-10`,
`atol=1e-12`.

### Method 3: WKB (for QNMs)

Implement the 3rd-order WKB formula (Schutz & Will 1985, Iyer & Will 1987).
Use as initial guesses for the QNM root-finding. The 6th-order Konoplya (2003)
extension improves accuracy significantly for small `l`.

### QNM root-finding

Locate QNMs as zeros of the Wronskian `W(ω) = Ψ_ext_in · dΨ_ext_out/dr* −
Ψ_ext_out · dΨ_ext_in/dr*` in the complex-`ω` plane. Use `scipy.optimize.root`
starting from WKB estimates.

---

## Output required

**Python code** (`main.py` plus logical modules). Every function must have a
docstring citing the equation it implements.

**Derivation document** (LaTeX `.tex` or Markdown `.md`): self-contained,
with every equation numbered and cited. Sections:
1. Background spacetime and shell — including the collapse problem and your
   chosen resolution.
2. Perturbation equations — RW and Zerilli, interior and exterior.
3. Interior solution — Riccati-Bessel, regularity (can cite `interior_ondas_cascara`).
4. Junction conditions — full derivation of `Δ` for your chosen model(s).
5. Numerical methods.
6. Results — with the validation checks listed above explicitly reported.

**Plots** (PDF + PNG, publication quality):
- Effective potentials `V(r*)` for both parities, `l = 2, 3, 4`.
- `R(ω)` and `T(ω)` vs `ωM` for both parities and junction models.
- Transfer-matrix vs. shooting comparison (must overlap).
- QNM frequencies in the complex-`ω` plane: shell system vs. pure Schwarzschild.
- QNM frequencies as a function of `R/M`.
- Energy conservation check: `|R + T − 1|` vs `ωM`.

**CSV table** of QNM frequencies: `l`, `n`, parity, model, `Re(ωM)`, `Im(ωM)`,
method, and a column flagging whether a literature comparison exists and what it gives.

---

## A note on the literature and honesty

If you search for references online and find a published paper that has already
computed some or all of what is asked here — **say so explicitly**. Reproduce
their results as a validation check, cite them properly, and then extend or
complement their work. The goal is correct physics, not novelty for its own sake.

If a step in the derivation is ambiguous or has multiple conventions in the
literature, flag it, explain the ambiguity, and state clearly which convention
you are using and why.

If the linearized junction conditions turn out to be more complex than the
schematic form `[dΨ/dr*] = Δ·Ψ(R)` — for example if even-parity perturbations
also displace the shell radially and couple the two sides nontrivially — say so
and handle it correctly, even if it means the scattering problem is a 2×2 system
rather than a scalar one.
