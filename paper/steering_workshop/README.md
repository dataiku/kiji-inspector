# Five-page steering workshop paper

This directory contains a self-contained workshop version of the causal SAE
steering study. It lives on branch steering-workshop-five-page in the separate
/home/shadeform/kiji-inspector-steering-workshop worktree.

- From Readable to Causal.tex is the manuscript.
- neurips_2026_formatting/neurips_2026.sty is the NeurIPS 2026 template.
- steering_refs.bib supplements the shared ../references.bib.
- images/ contains the figures used by the paper and appendix.
- build.sh compiles the PDF with local TeX or Docker.

Before submission, replace the placeholder in workshoptitle and verify whether
the target workshop uses dblblindworkshop or sglblindworkshop.

Build with:

~~~bash
./build.sh
~~~
