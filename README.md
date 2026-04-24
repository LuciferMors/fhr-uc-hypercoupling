# FHR–UC Phase Coupling in Severe Intrapartum Acidosis (CTU-UHB, n=506)

Analysis code and independent reimplementation accompanying the manuscript:

> **Early-labour fetal heart rate–uterine contraction phase coupling is elevated in severe intrapartum acidosis and is modulated by contraction frequency.** Shivhare RA. 2026.

Author: Rishi Arun Shivhare — `rishivhare07@gmail.com`

## What this repository contains

- `src/analysis.py` — original analysis pipeline (download → preprocessing → Hilbert phase coherence → statistics → figures).
- `src/repro.py` — independent reimplementation of the full pipeline, written from the Methods section of the manuscript alone, without reference to `analysis.py`.
- `requirements.txt` — Python dependencies.
- `results/results_features.csv` — per-record feature table (n=506).
- `results/results_summary.json` — aggregated statistical results.
- `figures/figure1_main.png`, `figure2_characterisation.png`, `figure3_permutation.png` — paper figures.
- `.zenodo.json` — Zenodo deposit metadata.
- `CITATION.cff` — GitHub citation metadata.

Raw CTG waveforms are **not** redistributed. They are available directly from PhysioNet (link below).

## Data source

CTU-UHB Intrapartum Cardiotocography Database — 552 intrapartum recordings from the University Hospital Brno, Czech Republic (2010–2012), published by Chudáček V, Spilka J, Burša M, et al. in BMC Pregnancy and Childbirth 2014;14:16.

- URL: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/
- Licence: Open Data Commons Attribution License v1.0 (ODC-By 1.0)
- No registration required.

## Reproducing the results

```bash
# 1. Clone and install
git clone https://github.com/LuciferMors/fhr-uc-hypercoupling.git
cd fhr-uc-hypercoupling
pip install -r requirements.txt

# 2. Download the CTU-UHB dataset (requires ~200 MB of disk)
#    Either let analysis.py do it, or use PhysioNet's wget command:
wget -r -N -c -np https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/ -P data/

# 3. Point both scripts at the download location (defaults to ./data/ctu_uhb):
export CTU_UHB_PATH="$PWD/data/physionet.org/files/ctu-uhb-ctgdb/1.0.0"
export FHR_OUT_DIR="$PWD/results"

# 4. Run
python src/analysis.py    # original pipeline
python src/repro.py        # independent reimplementation
```

Expected runtime: roughly 8 minutes on a laptop for each pipeline.

## Method in brief

Phase coherence between FHR and UC signals is computed using the Hilbert transform in sliding 5-minute windows (1-minute step):

1. Both signals are bandpass-filtered (0.01–1.0 Hz, third-order Butterworth).
2. Analytic signals are obtained via the Hilbert transform.
3. Instantaneous phase difference is computed sample-by-sample.
4. Window coherence is the magnitude of the mean complex unit phasor: `C = |mean(exp(i·Δφ))|`, which runs from 0 (random phase) to 1 (perfect phase locking).

Between-group differences are tested with Mann–Whitney U and validated with 2000-shuffle permutation tests. Contraction frequency is estimated by peak detection on the Savitzky–Golay smoothed UC signal.

## Headline results

| Group | n | Median coherence, min 0–30 [IQR] | p vs normal |
|---|---:|:--:|:--:|
| Normal (pH ≥ 7.20) | 347 | 0.250 [0.193–0.342] | — |
| Borderline (7.15–7.19) | 70 | 0.234 [0.207–0.325] | 0.61 |
| Moderate (7.05–7.14) | 64 | 0.239 [0.185–0.304] | 0.24 |
| **Severe (pH < 7.05)** | **25** | **0.336 [0.213–0.459]** | **0.018** |

Early-labour phase coherence is higher in severe-acidosis cases than in normal-outcome cases (minutes 10–15 aggregated: 0.349 vs 0.250, Mann–Whitney p = 0.005, permutation p = 0.0045). The effect is modulated by contraction frequency: amplified in labours with fewer than 4.4 contractions per 10 minutes, attenuated above that threshold.

The independent reimplementation (`src/repro.py`) reproduces the principal findings with stronger permutation validation: p = 0.0005 (minutes 10–15) and p = 0.0010 (minutes 15–20).

## Limitations

- Severe-acidosis group contains 25 cases; all findings are hypothesis-generating.
- Single-centre dataset (University Hospital Brno, Czech Republic).
- AUC of 0.648 for standalone detection is not sufficient for clinical screening.
- Intrapartum covariates such as oxytocin use, epidural analgesia, and maternal position are not coded in the dataset.

## Citation

If you use this code, please cite both the software archive and the dataset:

```
Shivhare RA. FHR-UC Phase Coupling Analysis (CTU-UHB). Zenodo. 2026.
   https://doi.org/10.5281/zenodo.XXXXXXX

Chudáček V, Spilka J, Burša M, et al. Open access intrapartum CTG database.
   BMC Pregnancy Childbirth. 2014;14:16.
```

Once the manuscript has a preprint DOI, add:

```
Shivhare RA. Early-labour fetal heart rate–uterine contraction phase coupling
   is elevated in severe intrapartum acidosis and is modulated by contraction
   frequency. medRxiv. 2026. https://doi.org/10.1101/2026.XX.XX.XXXXXXXX
```

## Licence

Code: MIT (see `LICENSE`).
Dataset: ODC-By 1.0 (PhysioNet). Attribution required separately.
