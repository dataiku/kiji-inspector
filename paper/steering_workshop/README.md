# Five-page steering workshop paper

This directory contains a self-contained workshop version of the causal SAE
steering study. It lives on branch steering-workshop-five-page in the separate
/home/shadeform/kiji-inspector-steering-workshop worktree.

- From Readable to Causal.tex is the manuscript.
- neurips_2026_formatting/neurips_2026.sty is the NeurIPS 2026 template.
- steering_refs.bib supplements the shared ../references.bib.
- images/ contains the figures used by the paper and appendix.
- artifacts/ holds every canonical battery output the numbers come from, so
  the paper is reproducible without the ignored 3.4 GB run tree; see its README.
- provenance.py regenerates artifacts/provenance.json: the pinned model
  revisions and per-file checksums behind those runs.
- health_inputs.py regenerates the compact dictionary-health inputs, so the
  health screen reproduces without the 478 MB activation captures.
- build.sh compiles the PDF with local TeX or Docker.

Targets the NeurIPS 2026 Workshop on Interpretability for Discovery.
Still unverified before submission: whether that workshop uses
dblblindworkshop or sglblindworkshop. The source currently sets the former.

Build with:

~~~bash
./build.sh
~~~
