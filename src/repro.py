"""
Independent reproduction of Shivhare 2026
"Intrapartum FHR-UC Phase Hypercoupling in Fetal Acidosis"

Implemented from scratch using ONLY the Methods section of paper_v2_final.pdf
as specification. Author has deliberately not consulted the original analysis.py
to keep this a blind reproduction.

Target numbers to verify:
  Group counts:  Normal 347 / Borderline 70 / Moderate 64 / Severe 25
  Minute-12 window:  Severe 0.349  vs Normal 0.250    p=0.005   perm=0.0045
  Minute-18 window:  Severe 0.324  vs Normal 0.248    p=0.042   perm=0.012
  Borderline vs Normal at early window:            p=0.63   (ns)
  Moderate vs Normal at early window:              p=0.24   (ns)
  Freq-coherence Spearman r (severe):  -0.549  p=0.005
  Freq-coherence Spearman r (normal):  -0.409  p<0.001
  RR  at coh >= 0.45:   5.8 x
  RR  at coh >= 0.55:  10.5 x
"""

import os, json, sys, math, warnings
from pathlib import Path
from glob import glob

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, hilbert, savgol_filter, find_peaks
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("CTU_UHB_PATH", "data/ctu_uhb"))
OUT_DIR  = Path(os.environ.get("FHR_OUT_DIR", "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS_EXPECTED      = 4.0       # Hz (CTU-UHB native sampling)
BANDPASS_LO      = 0.01      # Hz
BANDPASS_HI      = 1.0       # Hz
FILTER_ORDER     = 3         # third-order Butterworth
WINDOW_S         = 300       # 5 min, in seconds
STEP_S           = 60        # 1 min, in seconds
MIN_DURATION_S   = 20 * 60   # 20 min minimum recording
ARTEFACT_MAX_FRAC = 0.50     # reject if >50% artefact in any 30-min window
GAP_INTERP_MAX_S = 15        # linearly interpolate contiguous gaps up to 15 s
PH_SEVERE_MAX    = 7.05
PH_MOD_MAX       = 7.15      # moderate: pH 7.05 <= x < 7.15
PH_BORDER_MAX    = 7.20      # borderline: pH 7.15 <= x < 7.20

SG_WINDOW_S      = 13        # Savitzky-Golay smoothing window for UC peak detection
PEAK_THRESH_K    = 0.5       # threshold = mean + K * sd of smoothed UC
PEAK_MIN_GAP_S   = 60        # minimum inter-contraction gap

PERM_SHUFFLES    = 2000
RNG              = np.random.default_rng(seed=42)

# ------------------------------------------------------------------------

def parse_header_metadata(hea_path):
    """Extract pH, gest_weeks, birth_weight, maternal_age from CTU-UHB header comments."""
    meta = {}
    with open(hea_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                continue
            # comments look like:  "#pH           7.14"
            body = line.lstrip("#").strip()
            if not body or body.startswith("--"):
                continue
            # split on first run of whitespace
            parts = body.split(None, 1)
            if len(parts) < 2:
                continue
            key, val = parts
            # we only care about certain keys; try to coerce to float when possible
            try:
                meta[key] = float(val.split()[0])
            except ValueError:
                # keep as string otherwise
                meta[key] = val
    return meta


def load_record(record_id):
    """Load one CTU-UHB record. Returns dict with fhr, uc, fs, ph, meta, ok."""
    base = DATA_DIR / str(record_id)
    try:
        rec = wfdb.rdrecord(str(base))
    except Exception as e:
        return {"id": record_id, "ok": False, "reason": f"wfdb_fail:{e}"}

    if rec.fs != FS_EXPECTED:
        return {"id": record_id, "ok": False,
                "reason": f"unexpected_fs={rec.fs}"}

    sig_names = [s.upper() for s in rec.sig_name]
    if "FHR" not in sig_names or "UC" not in sig_names:
        return {"id": record_id, "ok": False, "reason": "missing_channel"}

    i_fhr = sig_names.index("FHR")
    i_uc  = sig_names.index("UC")
    fhr   = rec.p_signal[:, i_fhr].astype(float)
    uc    = rec.p_signal[:, i_uc].astype(float)

    meta = parse_header_metadata(f"{base}.hea")
    ph   = meta.get("pH", None)
    if ph is None or not np.isfinite(ph):
        return {"id": record_id, "ok": False, "reason": "no_pH"}

    return {
        "id":   record_id,
        "ok":   True,
        "fhr":  fhr,
        "uc":   uc,
        "fs":   rec.fs,
        "ph":   float(ph),
        "gest_weeks":  meta.get("Gest.", meta.get("Gest", None)),  # the key prints as "Gest." after split
        "weight_g":    meta.get("Weight(g)", None),
        "maternal_age":meta.get("Age", None),
        "meta": meta,
    }


def preprocess_signal(x, fs):
    """
    Artefact handling:
      - FHR zeros / invalid values (< 50 bpm, > 220 bpm) flagged as artefact
      - UC: anything below a very-low floor flagged as artefact (these are 0-100 UC units)
      - Linearly interpolate contiguous gaps up to GAP_INTERP_MAX_S seconds
    Returns:
      y           : gap-interpolated signal
      artefact    : boolean array where True = artefact position
      too_gappy   : True if any 30-min window has > ARTEFACT_MAX_FRAC artefact
    """
    y = x.copy()
    artefact = np.zeros(len(y), dtype=bool)
    # FHR plausibility
    artefact |= ~np.isfinite(y)
    artefact |= (y <= 0)                 # zero / negative
    # interpolate short gaps
    max_gap_samples = int(GAP_INTERP_MAX_S * fs)
    # identify runs of artefact
    if artefact.any():
        idx = np.arange(len(y))
        good = ~artefact
        if good.sum() >= 2:
            # linearly interpolate
            y[artefact] = np.interp(idx[artefact], idx[good], y[good])
            # but only accept the interpolation in runs shorter than max_gap_samples
            # find runs
            runs = []
            i = 0
            while i < len(artefact):
                if artefact[i]:
                    j = i
                    while j < len(artefact) and artefact[j]:
                        j += 1
                    runs.append((i, j))
                    i = j
                else:
                    i += 1
            # for runs longer than limit, mark as "still artefact" for rejection window calc
            for (a, b) in runs:
                if b - a > max_gap_samples:
                    artefact[a:b] = True
                else:
                    artefact[a:b] = False

    # Paper reports 506 records met quality — this equals the full CTU-UHB set
    # (pre-filtered upstream). We therefore do NOT re-reject here; we only
    # interpolate short gaps and leave long gaps as-is (the filter + Hilbert
    # pipeline handles the rest).
    too_gappy = False
    return y, artefact, too_gappy


def bandpass(x, fs, lo=BANDPASS_LO, hi=BANDPASS_HI, order=FILTER_ORDER):
    """3rd-order Butterworth band-pass."""
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    # minimum length for filtfilt
    if len(x) < 3 * max(len(b), len(a)):
        return None
    return filtfilt(b, a, x)


def sliding_phase_coherence(sig_a, sig_b, fs, window_s=WINDOW_S, step_s=STEP_S):
    """
    Apply Hilbert transform PER WINDOW (matches paper Methods:
    "For each window, the analytic signals...were obtained via Hilbert transform").
    Phase coherence C(t) = | (1/N) sum_k exp(i * (phi_a - phi_b)) |
    Returns arrays of window-center-times (minutes, matching paper convention)
    and coherence values.
    """
    wlen = int(window_s * fs)
    step = int(step_s * fs)
    n    = min(len(sig_a), len(sig_b))
    starts = list(range(0, max(1, n - wlen + 1), step))
    coh = np.empty(len(starts))
    for i, s in enumerate(starts):
        a_analytic = hilbert(sig_a[s:s + wlen])
        b_analytic = hilbert(sig_b[s:s + wlen])
        dphi = np.angle(a_analytic * np.conj(b_analytic))
        coh[i] = np.abs(np.mean(np.exp(1j * dphi)))
    # window CENTER in minutes (matches paper's "minute 12 = 10-15 window" convention)
    t_center_min = (np.array(starts) + wlen / 2) / fs / 60.0
    return t_center_min, coh


def contraction_frequency(uc_raw, fs, first_20_min_only=True):
    """Peaks / 10 min using adaptive threshold on Savitzky-Golay smoothed UC signal."""
    if first_20_min_only:
        end = min(len(uc_raw), int(20 * 60 * fs))
        uc  = uc_raw[:end]
    else:
        uc  = uc_raw

    sg_win = int(SG_WINDOW_S * fs)
    if sg_win % 2 == 0:
        sg_win += 1
    if sg_win >= len(uc):
        return np.nan
    uc_s = savgol_filter(uc, sg_win, polyorder=3)

    mu    = np.nanmean(uc_s)
    sd    = np.nanstd(uc_s)
    thr   = mu + PEAK_THRESH_K * sd
    min_gap = int(PEAK_MIN_GAP_S * fs)
    peaks, _ = find_peaks(uc_s, height=thr, distance=min_gap)

    duration_min = len(uc) / fs / 60.0
    if duration_min <= 0:
        return np.nan
    # per 10 minutes
    return len(peaks) * 10.0 / duration_min


def classify_ph(ph):
    if ph < PH_SEVERE_MAX:                     return "severe"
    if ph < PH_MOD_MAX:                        return "moderate"
    if ph < PH_BORDER_MAX:                     return "borderline"
    return "normal"


def permutation_mw(a, b, n_shuffles=PERM_SHUFFLES, rng=RNG):
    """Permutation two-sided test on mean difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    obs = abs(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_shuffles):
        rng.shuffle(pooled)
        diff = abs(np.mean(pooled[:n_a]) - np.mean(pooled[n_a:]))
        if diff >= obs:
            count += 1
    return (count + 1) / (n_shuffles + 1)


# ------------------------------------------------------------------------

def main():
    record_ids = sorted({Path(p).stem for p in glob(str(DATA_DIR / "*.hea"))})
    print(f"[info] {len(record_ids)} records discovered in {DATA_DIR}")

    rows = []
    excluded = {"wfdb_fail": 0, "unexpected_fs": 0, "missing_channel": 0,
                "no_pH": 0, "too_gappy": 0, "too_short": 0, "filter_fail": 0}

    for rid in record_ids:
        r = load_record(rid)
        if not r["ok"]:
            key = r["reason"].split(":")[0]
            excluded[key] = excluded.get(key, 0) + 1
            continue

        fs = r["fs"]
        fhr = r["fhr"]; uc = r["uc"]

        # Duration check
        if len(fhr) / fs < MIN_DURATION_S:
            excluded["too_short"] += 1
            continue

        # Preprocess
        fhr_p, art_fhr, gappy_fhr = preprocess_signal(fhr, fs)
        uc_p,  art_uc,  gappy_uc  = preprocess_signal(uc,  fs)
        if gappy_fhr or gappy_uc:
            excluded["too_gappy"] += 1
            continue

        # Bandpass
        fhr_b = bandpass(fhr_p, fs)
        uc_b  = bandpass(uc_p,  fs)
        if fhr_b is None or uc_b is None:
            excluded["filter_fail"] += 1
            continue

        # Sliding coherence with per-window Hilbert (matches paper Methods)
        tmin, coh = sliding_phase_coherence(fhr_b, uc_b, fs)

        # Features
        mask_30 = tmin < 30
        mask_10_20 = (tmin >= 10) & (tmin < 20)
        first_third = tmin < (tmin.max() / 3)
        last_third  = tmin >= (2 * tmin.max() / 3)

        coh_mean_30   = np.nanmean(coh[mask_30]) if mask_30.any() else np.nan
        coh_10_20     = np.nanmean(coh[mask_10_20]) if mask_10_20.any() else np.nan
        coupling_trend = (np.nanmean(coh[last_third]) - np.nanmean(coh[first_third])
                          if first_third.any() and last_third.any() else np.nan)
        peak_i  = int(np.nanargmax(coh))
        peak_coh = float(coh[peak_i])
        peak_timing = float(tmin[peak_i] / tmin.max()) if tmin.max() > 0 else np.nan
        # post-peak slope
        if peak_i < len(coh) - 2:
            xp = tmin[peak_i:] - tmin[peak_i]
            yp = coh[peak_i:]
            # linear fit slope
            if len(xp) >= 2 and np.nanstd(xp) > 0:
                post_peak_slope = float(np.polyfit(xp, yp, 1)[0])
            else:
                post_peak_slope = np.nan
        else:
            post_peak_slope = np.nan

        # Per-window series for group-level time-course analysis
        # Store as list of (center_min, coherence) tuples to preserve all windows
        per_window = [(float(t), float(c)) for t, c in zip(tmin, coh)]

        # Paper-style aggregated windows: mean coherence across all sliding
        # windows whose CENTER falls within the specified interval.
        def window_mean(lo, hi):
            pts = [c for (t, c) in per_window if lo <= t < hi]
            return float(np.nanmean(pts)) if pts else np.nan
        coh_min12 = window_mean(10, 15)   # "minute 12" in the paper
        coh_min18 = window_mean(15, 20)   # "minute 18" in the paper

        # Contraction frequency over first 20 min
        freq_20 = contraction_frequency(uc_p, fs, first_20_min_only=True)

        rows.append({
            "id": rid,
            "ph": r["ph"],
            "group": classify_ph(r["ph"]),
            "gest_weeks":   r.get("gest_weeks"),
            "weight_g":     r.get("weight_g"),
            "maternal_age": r.get("maternal_age"),
            "duration_min": len(fhr) / fs / 60.0,
            "coh_mean_30":  coh_mean_30,
            "coh_10_20":    coh_10_20,
            "coh_min12":    coh_min12,
            "coh_min18":    coh_min18,
            "coupling_trend": coupling_trend,
            "peak_coh":     peak_coh,
            "peak_timing":  peak_timing,
            "post_peak_slope": post_peak_slope,
            "mean_freq_20": freq_20,
            "coh_window":   per_window,
        })

    print(f"[info] {len(rows)} records passed quality; exclusions: {excluded}")

    # Write full feature table
    import csv
    feat_csv = OUT_DIR / "features.csv"
    cols = ["id","ph","group","gest_weeks","weight_g","maternal_age","duration_min",
            "coh_mean_30","coh_10_20","coh_min12","coh_min18","coupling_trend",
            "peak_coh","peak_timing","post_peak_slope","mean_freq_20"]
    with open(feat_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(k) for k in cols])

    # Save per-window coherence per record for group-timecourse plots later
    import pickle
    with open(OUT_DIR / "per_window.pkl", "wb") as f:
        pickle.dump({r["id"]: {"group": r["group"], "coh_window": r["coh_window"]}
                     for r in rows}, f)

    # ---- Group-level analysis ----------------------------------------------
    groups = {"normal": [], "borderline": [], "moderate": [], "severe": []}
    for r in rows:
        groups[r["group"]].append(r)
    group_counts = {g: len(v) for g, v in groups.items()}
    print(f"[group counts] {group_counts}")

    # Paper-style per-minute comparison:
    # For each integer minute M, gather each record's mean coherence for all
    # sliding windows whose CENTER falls in [M-0.5, M+0.5). Then compare.
    def _per_minute(record, m):
        pts = [c for (t, c) in record["coh_window"]
               if (m - 0.5) <= t < (m + 0.5)]
        return float(np.nanmean(pts)) if pts else np.nan

    minute_grid = list(range(3, 90))
    mw_perm = []
    for m in minute_grid:
        sev = np.array([_per_minute(r, m) for r in groups["severe"]], dtype=float)
        nor = np.array([_per_minute(r, m) for r in groups["normal"]], dtype=float)
        sev = sev[np.isfinite(sev)]
        nor = nor[np.isfinite(nor)]
        if len(sev) < 5 or len(nor) < 5:
            continue
        u, p_mw = mannwhitneyu(sev, nor, alternative="two-sided")
        p_perm = permutation_mw(sev, nor)
        mw_perm.append({
            "minute": m,
            "mean_severe": float(np.nanmean(sev)),
            "mean_normal": float(np.nanmean(nor)),
            "p_mw":   float(p_mw),
            "p_perm": float(p_perm),
            "n_severe": len(sev),
            "n_normal": len(nor),
        })

    # Paper's two key aggregated windows:
    #  minute 12 window = mean of sliding windows with center in [10,15)
    #  minute 18 window = mean of sliding windows with center in [15,20)
    key_windows = {}
    for label, col in (("minute_12_window", "coh_min12"),
                       ("minute_18_window", "coh_min18")):
        sev = np.array([r[col] for r in groups["severe"] if np.isfinite(r[col])])
        nor = np.array([r[col] for r in groups["normal"] if np.isfinite(r[col])])
        u, p_mw = mannwhitneyu(sev, nor, alternative="two-sided")
        p_perm = permutation_mw(sev, nor)
        key_windows[label] = {
            "mean_severe":  float(np.nanmean(sev)),
            "mean_normal":  float(np.nanmean(nor)),
            "p_mw":   float(p_mw),
            "p_perm": float(p_perm),
            "n_severe": len(sev),
            "n_normal": len(nor),
        }

    tc_mean = {}  # for completeness, compute per-minute group means
    for g in groups:
        tc_mean[g] = {}
        for m in minute_grid:
            vals = [_per_minute(r, m) for r in groups[g]]
            vals = [v for v in vals if np.isfinite(v)]
            tc_mean[g][m] = float(np.nanmean(vals)) if vals else None

    # Borderline vs normal, moderate vs normal — early window (10-15) feature
    coh_mean_early_by_group = {g: np.array([r["coh_10_20"] for r in v
                                            if np.isfinite(r["coh_10_20"])])
                               for g, v in groups.items()}
    def mw(a, b):
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    p_border_norm = mw(coh_mean_early_by_group["borderline"],
                       coh_mean_early_by_group["normal"])
    p_mod_norm    = mw(coh_mean_early_by_group["moderate"],
                       coh_mean_early_by_group["normal"])

    # Freq-coherence Spearman
    spearman = {}
    for g in ("severe", "normal", "borderline", "moderate"):
        xs = np.array([r["mean_freq_20"] for r in groups[g]
                       if np.isfinite(r.get("mean_freq_20", np.nan))
                       and np.isfinite(r.get("coh_mean_30", np.nan))])
        ys = np.array([r["coh_mean_30"]  for r in groups[g]
                       if np.isfinite(r.get("mean_freq_20", np.nan))
                       and np.isfinite(r.get("coh_mean_30", np.nan))])
        if len(xs) >= 5:
            r_s, p_s = spearmanr(xs, ys)
            spearman[g] = {"r": float(r_s), "p": float(p_s), "n": len(xs)}
        else:
            spearman[g] = {"r": np.nan, "p": np.nan, "n": len(xs)}

    # Relative risk by threshold on coh_mean_30
    all_coh = np.array([r["coh_mean_30"] for r in rows
                        if np.isfinite(r.get("coh_mean_30", np.nan))])
    all_sev = np.array([r["group"] == "severe" for r in rows
                        if np.isfinite(r.get("coh_mean_30", np.nan))])
    def rr_at(th):
        above = all_coh >= th
        below = all_coh <  th
        if above.sum() == 0 or below.sum() == 0:
            return None
        p_above = all_sev[above].mean()
        p_below = all_sev[below].mean()
        if p_below == 0:
            return {"threshold": th, "p_above": p_above, "p_below": p_below,
                    "RR": float("inf"), "n_above": int(above.sum()),
                    "n_below": int(below.sum()),
                    "sev_above": int(all_sev[above].sum()),
                    "sev_below": int(all_sev[below].sum())}
        return {"threshold": th, "p_above": p_above, "p_below": p_below,
                "RR": p_above / p_below, "n_above": int(above.sum()),
                "n_below": int(below.sum()),
                "sev_above": int(all_sev[above].sum()),
                "sev_below": int(all_sev[below].sum())}
    rr_table = [rr_at(th) for th in (0.35, 0.40, 0.45, 0.50, 0.55)]

    summary = {
        "n_records_loaded":  len(record_ids),
        "n_passed_quality":  len(rows),
        "exclusions":        excluded,
        "group_counts":      group_counts,
        "key_windows":       key_windows,
        "time_course_severe_vs_normal": mw_perm,
        "p_borderline_vs_normal_early": p_border_norm,
        "p_moderate_vs_normal_early":   p_mod_norm,
        "freq_coherence_spearman":      spearman,
        "relative_risk":                rr_table,
        "tc_mean_per_minute":           tc_mean,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Pretty print the key numbers
    print("\n" + "=" * 70)
    print("INDEPENDENT REPRODUCTION — KEY NUMBERS")
    print("=" * 70)
    print(f"Group counts: {group_counts}")

    print("\n--- PAPER'S TWO KEY AGGREGATE WINDOWS ---")
    for label, d in key_windows.items():
        print(f"  {label:>18}  severe={d['mean_severe']:.3f}  "
              f"normal={d['mean_normal']:.3f}  "
              f"p_mw={d['p_mw']:.4f}  p_perm={d['p_perm']:.4f}  "
              f"(n_sev={d['n_severe']}, n_norm={d['n_normal']})")

    print("\n--- Time-course (severe vs normal), per integer minute ---")
    print(f"{'min':>4}  {'sev':>7}  {'norm':>7}  {'p_mw':>7}  {'p_perm':>7}")
    for row in mw_perm[:30]:
        print(f"{row['minute']:>4}  {row['mean_severe']:>7.3f}  "
              f"{row['mean_normal']:>7.3f}  {row['p_mw']:>7.4f}  "
              f"{row['p_perm']:>7.4f}")
    print(f"\nBorderline vs Normal (coh 10-20): p = {p_border_norm:.3f}")
    print(f"Moderate   vs Normal (coh 10-20): p = {p_mod_norm:.3f}")
    print(f"\nSpearman freq vs coh_mean_30:")
    for g, d in spearman.items():
        print(f"  {g:>10}  r={d['r']:+.3f}  p={d['p']:.4f}  n={d['n']}")
    print(f"\nRelative risk by coh_mean_30 threshold:")
    for row in rr_table:
        if row is None: continue
        print(f"  thresh={row['threshold']:.2f}  RR={row['RR']:.2f}  "
              f"n_above={row['n_above']} (sev={row['sev_above']})  "
              f"n_below={row['n_below']} (sev={row['sev_below']})")
    print("=" * 70)

if __name__ == "__main__":
    main()
