# Hopfield class reading

This bundle accompanies **Chapter 5: Memory as an Energy
Landscape---Hopfield** in *NeuroAI: Theoretical Foundations of Dynamics, Learning and
Computation in Brains, Minds, and Machines*.

The manuscript is formatted as an AIP REVTeX 4.2 reprint (two columns). It
develops the 1982 and 1984 PNAS models, the Amit--Gutfreund--Sompolinsky
statistical mechanics, optimization, polynomial and exponential dense
associative memory, and the modern Hopfield/attention relation.

## Contents

- *hopfield_book_chapter.tex*: manuscript source
- *hopfield_book_chapter.pdf*: compiled chapter
- *hopfield_book_chapter_references.bib*: bibliography
- *code/hopfield_replication.py*: seeded replication code
- *numerical_results.json*: complete numerical output from the full run
- *figures/*: every figure in PDF, PNG, and SVG
- *requirements.txt*: Python dependencies

## Reproduce the figures

From this directory:

    python -m pip install -r requirements.txt
    python code/hopfield_replication.py \
      --output figures \
      --results numerical_results.json

The default seed is 1982. Add the *--quick* flag to reduce Monte Carlo
repetitions. No external data or GPU is required.

## Compile the manuscript

With a TeX installation containing REVTeX 4.2:

    ./compile_chapter.sh

The bundle includes the REVTeX 4.2 class and AIP bibliography style under
*vendor/revtex*, so the script also works when they are absent from the default
TeX installation. Alternatively, with a system-wide REVTeX installation run
`latexmk -pdf hopfield_book_chapter.tex` directly.

The class declaration is:

    \documentclass[aip,cha,reprint,amsmath,amssymb,longbibliography]{revtex4-2}

## Numerical landmarks from the full run

- Classical recall: \(N=400\), 18 memories, initial overlap \(0.44\), final
  overlap \(1.0\).
- Zero-temperature continuation: terminal retrieval load
  \(\alpha \approx 0.137596\), compared with the theoretical
  \(\alpha_c \simeq 0.138\).
- Graded response: target overlap increases from \(0.408\) to \(0.966\); the
  largest sampled energy increase is \(2.84\times10^{-14}\), i.e. numerical
  roundoff.
- Polynomial dense memory: exact energy-difference stability tests accompany
  the analytic \(K_{\max}\sim N^{n-1}/\ln N\) curves.
- Modern continuous memory: the JSON records one-step retrieval as a function
  of memory count and inverse temperature, together with fixed-point energy.

Each simulation is deliberately synthetic and isolates one mechanism. The
manuscript states which conclusions are numerical illustrations and which
require the analytic thermodynamic-limit arguments.

## August 2026 revision

This revision repairs Eqs. (11) and (49), adds a reader's map for the
replica-symmetric reduction, clarifies the small RSB shift of the classical
capacity boundary and Demircigil et al.'s exponential-rate condition, moves a
model-versus-brain scope checkpoint earlier in the chapter, and expands the
explanations around several dense derivations. Figure 2's 28% corruption and
initial overlap of 0.44 are intentionally unchanged.
