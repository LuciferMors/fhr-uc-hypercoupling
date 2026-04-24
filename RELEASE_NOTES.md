# v1.0.0 — Manuscript-associated release

First public release of the analysis code accompanying:

> **Early-labour fetal heart rate–uterine contraction phase coupling is elevated in severe intrapartum acidosis and is modulated by contraction frequency.** Shivhare RA. 2026.

## Contents

- `src/analysis.py` — original analysis pipeline.
- `src/repro.py` — independent reimplementation from the Methods section alone.
- `requirements.txt` — Python dependencies.
- `results/results_features.csv` — per-record feature table (n=506).
- `results/results_summary.json` — aggregated statistical results.
- `figures/figure1_main.png`, `figure2_characterisation.png`, `figure3_permutation.png`.
- `LICENSE` (MIT), `CITATION.cff`, `.zenodo.json`.

## Reproducing the results

See `README.md`. All results reproduce from the public CTU-UHB database with a single pipeline call per script; expected runtime is approximately 8 minutes on a laptop.

## Not included

The CTU-UHB raw waveforms are distributed by PhysioNet under the Open Data Commons Attribution License v1.0 and are not redistributed here. Download them directly from https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/.

## Citation

```
Shivhare RA. FHR-UC Phase Coupling Analysis (CTU-UHB). Zenodo. 2026.
Chudáček V, Spilka J, Burša M, et al. Open access intrapartum CTG database.
    BMC Pregnancy Childbirth. 2014;14:16.
```
