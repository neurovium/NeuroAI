# Chapter 2: Neural Logic, Invariance, and the Retina

Complete reproducibility package for the McCulloch--Pitts book chapter in
*NeuroAI: Theoretical Foundations of Dynamics, Learning and Computation in
Brains, Minds, and Machines*.

## Contents

- `mcculloch_book_chapter.tex` -- AIP REVTeX two-column chapter source
- `mcculloch_book_chapter.pdf` -- compiled chapter
- `mcculloch_book_chapter_references.bib` -- chapter-scoped bibliography
- `code/mcculloch_replication.py` -- deterministic code for all seven figures
- `figures/` -- every figure in PDF, PNG, and editable SVG
- `numerical_results.json` -- truth tables, state-transition data, model
  parameters, and quantitative recovery checks
- `vendor/revtex/` -- portable REVTeX 4.2 class and AIP bibliography styles
- `requirements.txt` -- Python dependencies
- `Makefile` -- figure generation, compilation, and verification commands

The figures are original schematics or controlled pedagogical demonstrations;
they are not digitizations of historical figures. The retinal channels are a
qualitative effective model and are not fitted to the 1959 recordings. The four
modeled channels correspond to the four principal operations emphasized by
Lettvin and colleagues; the rare fifth absolute-darkness group is discussed in
the chapter but is not promoted to a fifth modeled channel.

## Reproduce everything

From this directory:

```bash
python code/mcculloch_replication.py \
  --output figures \
  --results numerical_results.json

make pdf
make verify
```

Or run the complete target:

```bash
make all
```

The default random seed is `1943`. Each figure is written as vector PDF,
editable SVG, and raster PNG. The supplied REVTeX files make the LaTeX build
independent of a system-wide REVTeX installation.

## Numerical checks

The supplied run establishes the following deterministic checks:

- AND, OR, NOT, and the two-stage XOR truth tables are exact.
- The three-bit recurrent map has a fixed point, a period-three attractor, and
  genuine transients. The trajectory from `101` has transient length
  `mu = 3` and cycle length `lambda = 3`.
- Transformation-orbit pooling raises mean normalized template match from
  approximately `0.270` to `0.717` in the specified finite test.
- Feedback canonicalization reduces absolute pose error from `0.56` to about
  `0.0038`.
- The retinal stimulus contains an approximately 30-fold illumination ramp.
- With 120,000 seeded stimulus samples, the linear STA/filter correlation is
  about `0.990`, the energy-model STA correlation is about `0.061`, and the
  leading STC direction recovers the energy filter with correlation about
  `0.989`.

Exact values and all state transitions are recorded in
`numerical_results.json`.
