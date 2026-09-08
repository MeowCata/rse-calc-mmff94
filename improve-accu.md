# Accuracy and uncertainty roadmap

The production model currently uses MMFF94, exhaustive single-bond ring
opening for substituted rings, adaptive ETKDG plus ring-pucker seeds, and the
validated Monte Carlo/parallel-tempering torsion search. The following ideas
could improve accuracy or quantify uncertainty, but are deliberately not
implemented in this change because they require new validation data or would
alter the calibrated model.

## Higher-fidelity energy checks

- Re-optimize the lowest MMFF conformers with a dispersion-aware DFT method
  (for example a short-range screened hybrid plus D3 correction) and fit a
  ring-size/substituent correction against measured heats of formation.
- Compare MMFF94 with an independent force field (such as GAFF2 or OPLS) and
  use the inter-model spread as an uncertainty component rather than silently
  choosing one result.
- Replace the substituted-ring H-capped opening reference with a strictly
  bond-balanced homodesmotic reaction. This is especially important for
  heavily crowded small rings, where cyclic and acyclic vdW repulsion can
  cancel and even produce negative raw strain.

## Sampling and uncertainty

- Run independent random seeds and report the minimum/mean/spread of the
  converged cyclic and acyclic energies. The current deterministic seed is
  reproducible, but it cannot expose missed conformational basins.
- Use an adaptive stopping rule based on repeated recovery of the same lowest
  energy and a confidence interval on the Boltzmann average, instead of only
  fixed MC/PT step budgets.
- Preserve multiple low-energy ring puckers through the full reference path and
  propagate conformer weights into the geometry diagnostics, which are now
  snapshots from one representative conformer.

## Experimental calibration

- Expand the reference set with matched substitution patterns and explicit
  cis/trans measurements. Fit robust per-size regressions with held-out
  validation and bootstrap confidence intervals, rather than relying only on
  the current linear fit uncertainty.
- Add uncertainty flags for low MMFF parameter coverage, extreme bulk scores,
  and raw strain/decomposition sign reversals. These cases should remain
  supported for compatibility but be clearly marked as extrapolations.

These proposals should be evaluated against the existing regression and bulky
benchmarks before being enabled in the default pipeline.
