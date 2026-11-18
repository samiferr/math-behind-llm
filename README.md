# The Math Behind Large Language Models

*A step-by-step beginner's guide, from arithmetic to transformers.*

**Current edition: First Edition, version 1.0.0 (November 2026).**

This book explains the math behind large language models (LLMs) to readers who
only know school arithmetic. It goes one step at a time from functions and graphs
to attention and transformers. Each idea starts with a real-world example, then
gives the intuition, a chart or diagram, the formula, and where the idea is used
inside an LLM.

## Download

| Edition | Version | PDF | EPUB |
| --- | --- | --- | --- |
| First Edition | 1.0.0 | [PDF](editions/1st-edition/the-math-behind-llms_1st-edition_v1.0.0.pdf) | [EPUB](editions/1st-edition/the-math-behind-llms_1st-edition_v1.0.0.epub) |

Each tagged version is also published on the repository's **Releases** page by
the build workflow.

## Contents

| Part | Chapters |
| --- | --- |
| Preface | Who the book is for, how it teaches, the roadmap |
| I. Foundations | 1. What a language model really does · 2. Numbers, functions and graphs |
| II. Linear algebra | 3. Vectors · 4. The dot product · 5. Matrices · 6. Tokens and embeddings |
| III. Probability | 7. Probability basics · 8. Distributions and softmax · 9. Surprise, entropy and loss |
| IV. Calculus and learning | 10. Derivatives · 11. Gradients and the chain rule · 12. Gradient descent · 13. Neural networks from scratch |
| V. The transformer | 14. Attention · 15. The transformer block · 16. Training an LLM · 17. Generating text · 18. Putting it all together |
| Appendices | Notation, key formulas, glossary, answers to the exercises, further reading |

## Repository layout

```
manuscript/        The book text, one Markdown file per chapter (the source of truth)
figures/           The 44 charts and diagrams, named fig-CC-NN.png (chapter, number)
cover/cover.html   Cover design; the title, author and edition label are filled in at build time
styles/            LaTeX settings for the PDF and CSS for the EPUB
templates/         Pandoc LaTeX template
scripts/build.py   Builds the cover, the EPUB and the PDF
scripts/new_edition.py  Starts a new edition, update or revision
metadata.yaml      Title, author, edition number, version, date and edition history
editions/          The published files for each edition
CHANGELOG.md       What changed in each version
```

## Editions and versions

Version numbers follow **EDITION.UPDATE.REVISION**:

| Change | Command | Example | Cover and title page |
| --- | --- | --- | --- |
| New edition (major rewrite, new chapters) | `make new-edition NOTE="..."` | 1.2.1 → 2.0.0 | "Second Edition" |
| Update within an edition (new sections, figures) | `make new-update NOTE="..."` | 1.0.0 → 1.1.0 | unchanged |
| Revision (corrections, typos) | `make new-revision NOTE="..."` | 1.0.0 → 1.0.1 | unchanged |

The command updates `metadata.yaml` (edition, version, date and the edition
history) and adds an entry to `CHANGELOG.md`. The edition label on the cover, the
title page, the copyright page's edition history and the output file names all
come from `metadata.yaml`, so nothing else needs editing by hand.

To release a new version:

```bash
make new-edition NOTE="Adds chapters on mixture of experts and reasoning models"
# ...edit manuscript/ and figures/...
make publish                  # builds and copies files into editions/2nd-edition/
git add -A && git commit -m "Second edition"
git tag v2.0.0 && git push && git push --tags   # the workflow publishes a Release
```

Earlier editions stay in `editions/`, so readers can always find the version they own.

## Building locally

Requirements: Python 3 with PyYAML, pandoc 3.x, XeLaTeX (TeX Live with
`texlive-latex-extra`), qpdf, Google Chrome or Chromium (to render the cover),
and the fonts TeX Gyre Pagella, Inter and DejaVu.

```bash
make all        # writes dist/<name>.pdf and dist/<name>.epub
make pdf        # just the PDF
make epub       # just the EPUB
```

The EPUB uses MathML for formulas. Apple Books, Thorium and most EPUB 3 readers
display them; older readers may show them as plain text. The PDF typesets every
formula.

## Notes

Figures labelled "illustrative" use realistic made-up values chosen to teach a
pattern; they are not measurements from a specific model.

Copyright © 2026 Sam. All rights reserved.
