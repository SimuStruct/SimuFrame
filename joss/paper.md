---
title: 'SimuFrame: A Python package for structural analysis'
tags:
  - Python
  - finite element method
  - geometric nonlinearity
  - total langragian
  - structural engineering
authors:
  - name: Alysson B. Barros
    orcid: 0009-0008-1880-6615
    equal-contrib: true
    affiliation: 1
  - name: Jordlly R. B. Silva
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: Federal Rural University of Pernambuco, Academic Unit of Cabo de Santo Agostinho (UFRPE/UACSA)
   index: 1
   ror: 00hx57361
 - name: Dept. of Civil and Environmental Engineering, Federal University of Pernambuco (UFPE)
   index: 2
date: 13 January 2026
bibliography: paper.bib
---

# Summary

`SimuFrame` is a Python-based finite element analysis (FEA) library designed
for the nonlinear analysis of three-dimensional space frames. It implements
both Timoshenko and Euler-Bernoulli beam theories, as well as truss analyses.
The nonlinear analysis is based on a Total Lagrangian formulation, where the
structure is all referenced in the initial (undeformed) configuration. This
formulation makes the software particularly adequate for the analysis of thin-walled
structures and slender frames where second-order effects are critical, while
maintaining the hypothesis of small strains and moderate displacements.

The core solver utilizes either a Newton-Raphson or an Arc-Length method
to solve nonlinear equilibrium equations, enabling the tracking of post-critical
equilibrium. `SimuFrame` also includes a module for eigenvalue buckling analysis
to assess structural stability. For post-processing, the software leverages `PyVista`
and `Matplotlib` to generate intuitive visualization of deformed configurations and 
internal forces distributions, streamlining the interpretation of structural behavior.

# Statement of need

In civil and structural engineering, accounting for geometric nonlinearity
is essential for the safe design of slender structures. While established
commercial solvers like Ansys and Abaqus offer powerful capabilities for
these analyses, they are often proprietary, computationally expensive, and
lack the flexibility required for rapid prototyping or educational exploration
of finite element algorithms.

`SimuFrame` addresses this need by providing an open-source, accessible, and
transparent implementation of nonlinear beam theory. It allows researchers
and engineers to inspect, modify, and extend the underlying formulations for
specific use cases. The software has been validated against industry standards;
benchmark tests compare `SimuFrame` against Abaqus and Dlubal RFEM, yielding
displacement results within a 1.5% discrepancy in complex scenarios and matching
theoretical values in canonical cases.

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
