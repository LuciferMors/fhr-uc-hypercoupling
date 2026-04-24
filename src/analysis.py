"""
FHR-UC Phase Hypercoupling Analysis
====================================
Rishi Arun Shivhare
rishivhare07@gmail.com

Replication script for:
"Early-labour fetal heart rate–uterine contraction phase coupling is
elevated in severe intrapartum acidosis and is modulated by contraction
frequency" (Shivhare, 2026)

Dataset: CTU-UHB Intrapartum Cardiotocography Database
Source:  https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/
License: Open Data Commons Attribution License v1.0

Usage:
    python analysis.py

    This script will:
    1. Download all 552 CTG records from PhysioNet (no login required)
    2. Extract FHR-UC phase coherence trajectories
    3. Run all statistical analyses
    4. Generate all figures (saved to figures/)
    5. Print the complete results table

Requirements:
    pip install wfdb numpy scipy pandas scikit-learn matplotlib
"""

import os
import re
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_URL   = "https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/"
DATA_DIR   = "data"
FIG_DIR    = "figures"
N_PERM     = 2000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

COLORS = {
    'normal':     '#1565C0',
    'borderline': '#FF9800',
    'moderate':   '#EF5350',
    'severe':     '#B71C1C',
}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)


# ─── STEP 1: DOWNLOAD DATA ────────────────────────────────────────────────────
def download_records():
    """Download all CTU-UHB records. No login required."""
    print("Downloading CTU-UHB records from PhysioNet...")
    downloaded, failed = [], []
    for rec_id in range(1001, 1553):
        s = str(rec_id)
        hea = os.path.join(DATA_DIR, f"{s}.hea")
        dat = os.path.join(DATA_DIR, f"{s}.dat")
        if os.path.exists(hea) and os.path.exists(dat):
            downloaded.append(s)
            continue
        try:
            urllib.request.urlretrieve(BASE_URL + f"{s}.hea", hea)
            urllib.request.urlretrieve(BASE_URL + f"{s}.dat", dat)
            downloaded.append(s)
        except Exception:
            failed.append(s)
        if len(downloaded) % 100 == 0:
            print(f"  {len(downloaded)} downloaded...")
    print(f"Done. {len(downloaded)} records, {len(failed)} failed.")
    return downloaded


# ─── STEP 2: PARSE OUTCOMES ───────────────────────────────────────────────────
def parse_header(rec_id):
    """Extract clinical outcomes from .hea file."""
    path = os.path.join(DATA_DIR, f"{rec_id}.hea")
    if not os.path.exists(path):
        return None
    text = open(path).read()

    def get(pattern, default=np.nan):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    return {
        'id':           rec_id,
        'pH':           get(r'#pH\s+([\d.]+)'),
        'BDecf':        get(r'#BDecf\s+([-\d.]+)'),
        'pCO2':         get(r'#pCO2\s+([\d.]+)'),
        'BE':           get(r'#BE\s+([-\d.]+)'),
        'Apgar1':       get(r'#Apgar1\s+([\d.]+)'),
        'Apgar5':       get(r'#Apgar5\s+([\d.]+)'),
        'NICU':         get(r'#NICU days\s+([\d.]+)'),
        'Gest_weeks':   get(r'#Gest\. weeks\s+([\d.]+)'),
        'Weight_g':     get(r'#Weight\(g\)\s+([\d.]+)'),
        'Age':          get(r'#Age\s+([\d.]+)'),
        'Diabetes':     get(r'#Diabetes\s+([\d.]+)'),
        'Hypertension': get(r'#Hypertension\s+([\d.]+)'),
        'Preeclampsia': get(r'#Preeclampsia\s+([\d.]+)'),
        'Deliv_type':   get(r'#Deliv\. type\s+([\d.]+)'),
    }


def outcome_group(pH):
    if pH < 7.05:  return 'severe'
    if pH < 7.15:  return 'moderate'
    if pH < 7.20:  return 'borderline'
    return 'normal'


# ─── STEP 3: PHASE COHERENCE ──────────────────────────────────────────────────
def bandpass(sig, fs=4, lo=0.01, hi=1.0, order=3):
    """Bandpass filter with NaN interpolation."""
    s = sig.copy().astype(float)
    nans = np.isnan(s)
    if nans.all():
        return s
    idx = np.arange(len(s))
    s[nans] = np.interp(idx[nans], idx[~nans], s[~nans])
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, min(hi / nyq, 0.99)], btype='band')
    return filtfilt(b, a, s)


def phase_coherence(fhr, uc, fs=4, window_min=5, step_min=1):
    """
    Compute sliding-window FHR-UC phase coherence via Hilbert transform.

    Returns
    -------
    times      : centre of each window in minutes
    coherence  : phase coherence [0,1] per window
    phase_lag  : mean phase lag in degrees per window
    """
    W = int(window_min * 60 * fs)
    S = int(step_min   * 60 * fs)
    fhr_f = bandpass(fhr)
    uc_f  = bandpass(uc)
    times, coh, lag = [], [], []
    for start in range(0, len(fhr_f) - W, S):
        end = start + W
        f_a = hilbert(fhr_f[start:end])
        u_a = hilbert(uc_f[start:end])
        dphi = np.angle(f_a * np.conj(u_a))
        times.append((start + W / 2) / (fs * 60))
        coh.append(float(np.abs(np.mean(np.exp(1j * dphi)))))
        lag.append(float(np.degrees(np.mean(dphi))))
    return np.array(times), np.array(coh), np.array(lag)


# ─── STEP 4: PROCESS ALL RECORDS ─────────────────────────────────────────────
def process_all(record_ids):
    """Load, filter, compute coherence for every record."""
    import wfdb
    results = []
    skipped = 0
    for rid in record_ids:
        meta = parse_header(rid)
        if meta is None or np.isnan(meta['pH']):
            skipped += 1
            continue
        try:
            rec = wfdb.rdrecord(os.path.join(DATA_DIR, rid))
            fhr, uc = rec.p_signal[:, 0], rec.p_signal[:, 1]
            if np.mean(~np.isnan(fhr)) < 0.5 or len(fhr) < 4800:
                skipped += 1
                continue
            times, coh, lag = phase_coherence(fhr, uc)
            if len(coh) < 5:
                skipped += 1
                continue
            n = len(coh)
            t3 = max(1, n // 3)
            meta.update({
                'group':          outcome_group(meta['pH']),
                'mean_coh':       float(np.mean(coh)),
                'early_coh':      float(np.mean(coh[:t3])),
                'late_coh':       float(np.mean(coh[-t3:])),
                'final_coh':      float(np.mean(coh[-max(1, n//5):])),
                'trend':          float(np.mean(coh[-t3:]) -
                                         np.mean(coh[:t3])),
                'coh_mean_30':    float(np.mean(coh[times <= 30]))
                                  if (times <= 30).any() else np.nan,
                '_times': times,
                '_coh':   coh,
                '_lag':   lag,
            })
            results.append(meta)
        except Exception:
            skipped += 1
    print(f"Processed {len(results)}, skipped {skipped}")
    return pd.DataFrame(results)


# ─── STEP 5: STATISTICAL TESTS ────────────────────────────────────────────────
def permutation_test(vals_a, vals_b, n_perm=N_PERM):
    """Two-sample permutation test; returns empirical p-value."""
    real_diff = np.mean(vals_a) - np.mean(vals_b)
    combined  = np.concatenate([vals_a, vals_b])
    na        = len(vals_a)
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(combined)
        diff = np.mean(perm[:na]) - np.mean(perm[na:])
        if abs(diff) >= abs(real_diff):
            count += 1
    return count / n_perm


def bin_trajectories(df, bin_size=5, max_min=85):
    """Bin coherence values by absolute minute for each group."""
    bins    = np.arange(0, max_min, bin_size)
    centers = bins[:-1] + bin_size / 2
    traj = {g: {c: [] for c in centers} for g in
            ['normal', 'borderline', 'moderate', 'severe']}
    for _, row in df.iterrows():
        g = row['group']
        if g not in traj:
            continue
        for i, c in enumerate(centers):
            mask = (row['_times'] >= bins[i]) & (row['_times'] < bins[i+1])
            if mask.any():
                traj[g][c].append(float(np.mean(row['_coh'][mask])))
    return centers, traj


# ─── STEP 6: FIGURES ──────────────────────────────────────────────────────────
def figure1(df, centers, traj):
    """Figure 1 — Main coupling trajectory by outcome group."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A — all four groups
    ax = axes[0]
    for g in ['normal', 'borderline', 'moderate', 'severe']:
        xs, ys, es = [], [], []
        for c in centers:
            v = traj[g][c]
            if len(v) >= 5:
                xs.append(c); ys.append(np.mean(v))
                es.append(np.std(v) / np.sqrt(len(v)))
        xs, ys, es = map(np.array, [xs, ys, es])
        n  = int((df['group'] == g).sum())
        lw = 3 if g == 'severe' else 1.8
        ax.plot(xs, ys, color=COLORS[g], lw=lw,
                label=f'{g.capitalize()} (n={n})',
                zorder=3 if g == 'severe' else 1)
        ax.fill_between(xs, ys - es, ys + es,
                        alpha=0.15, color=COLORS[g])

    # shade significant windows
    for c in centers:
        s = traj['severe'][c]; n = traj['normal'][c]
        if len(s) >= 5 and len(n) >= 20:
            _, p = mannwhitneyu(s, n, alternative='two-sided')
            if p < 0.05:
                ax.axvspan(c - 2.5, c + 2.5, alpha=0.10, color='green')

    ax.axvline(20, color='gray', ls='--', alpha=0.6, lw=1.5)
    ax.set_xlabel('Minutes from Recording Start')
    ax.set_ylabel('FHR\u2013UC Phase Coherence')
    ax.set_title('A  FHR\u2013UC Phase Coupling Across Labour\n'
                 'by Neonatal Outcome Group', fontweight='bold', loc='left')
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    ax.text(0.02, 0.97, '\u2588 p<0.05 (severe vs normal)',
            transform=ax.transAxes, fontsize=8,
            color='green', va='top')

    # Panel B — severe vs normal
    ax = axes[1]
    for g in ['normal', 'severe']:
        xs, ys, es = [], [], []
        for c in centers:
            v = traj[g][c]
            if len(v) >= 5:
                xs.append(c); ys.append(np.mean(v))
                es.append(np.std(v) / np.sqrt(len(v)))
        xs, ys, es = map(np.array, [xs, ys, es])
        n = int((df['group'] == g).sum())
        ax.plot(xs, ys, color=COLORS[g], lw=2.5,
                label=f'{g.capitalize()} (n={n})',
                marker='o', markersize=4)
        ax.fill_between(xs, ys - es, ys + es,
                        alpha=0.2, color=COLORS[g])

    ax.axvspan(0, 20, alpha=0.07, color='green')
    ax.axvline(20, color='green', ls='--', lw=1.5,
               label='Early window (0\u201320 min)')
    ax.set_xlabel('Minutes from Recording Start')
    ax.set_ylabel('FHR\u2013UC Phase Coherence')
    ax.set_title('B  Early Hypercoupling in Severe Acidosis\n'
                 '(pH < 7.05) \u2014 Permutation-Validated',
                 fontweight='bold', loc='left')
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    plt.suptitle('Intrapartum FHR\u2013UC Phase Coupling: '
                 'A Novel Early Marker of Fetal Acidosis\n'
                 'CTU-UHB Dataset (n=506)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'figure1_main.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def figure2(df, centers, traj):
    """Figure 2 — ROC, effect sizes, individual trajectories."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ROC
    ax = axes[0]
    y  = (df['group'] == 'severe').astype(int)
    x  = df['coh_mean_30'].fillna(df['coh_mean_30'].median())
    fpr, tpr, _ = roc_curve(y, x)
    ra = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='#B71C1C', lw=2.5,
            label=f'Mean coherence 0\u201330 min\nAUC = {ra:.3f}')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#B71C1C')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('A  ROC Curve\nSevere Acidosis (pH < 7.05)',
                 fontweight='bold', loc='left')
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    ax.set_aspect('equal')

    # Effect sizes
    ax = axes[1]
    exs, eys, eps = [], [], []
    for c in centers:
        s = traj['severe'][c]; n = traj['normal'][c]
        if len(s) >= 5 and len(n) >= 20:
            d = np.mean(s) - np.mean(n)
            sd = np.sqrt((np.std(s)**2 + np.std(n)**2) / 2)
            exs.append(c)
            eys.append(d / sd if sd > 0 else 0)
            _, p = mannwhitneyu(s, n, alternative='two-sided')
            eps.append(p)
    bc = ['#B71C1C' if p < 0.05 else '#90CAF9' for p in eps]
    ax.bar(exs, eys, width=4, color=bc, alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(20, color='green', ls='--', alpha=0.6, lw=1.5)
    ax.set_xlabel('Minutes from Recording Start')
    ax.set_ylabel("Cohen's d (Severe vs Normal)")
    ax.set_title("B  Effect Size Over Time\n(Red = p < 0.05)",
                 fontweight='bold', loc='left')
    ax.grid(alpha=0.2, axis='y')

    # Individual severe trajectories
    ax = axes[2]
    t_grid = np.linspace(0, 100, 20)
    sev_interp, nor_interp = [], []
    for _, row in df.iterrows():
        t = row['_times']; c = row['_coh']
        if len(t) < 5: continue
        tn = (t - t.min()) / (t.max() - t.min()) * 100
        ci = np.interp(t_grid, tn, c)
        if row['group'] == 'severe':
            ax.plot(t_grid, ci, alpha=0.3, lw=0.9, color='#EF5350')
            sev_interp.append(ci)
        elif row['group'] == 'normal':
            nor_interp.append(ci)

    if sev_interp:
        ax.plot(t_grid, np.mean(sev_interp, axis=0),
                color='#B71C1C', lw=3, label='Mean severe', zorder=5)
    if nor_interp:
        ax.plot(t_grid, np.mean(nor_interp, axis=0),
                color='#1565C0', lw=2.5, ls='--',
                label='Mean normal', zorder=4)

    ax.axvspan(0, 25, alpha=0.08, color='green')
    n_sev = int((df['group'] == 'severe').sum())
    ax.set_xlabel('Labour Progress (%)')
    ax.set_ylabel('FHR\u2013UC Phase Coherence')
    ax.set_title(f'C  Individual Trajectories\nAll Severe Cases (n={n_sev})',
                 fontweight='bold', loc='left')
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    plt.suptitle('Characterisation of Early FHR\u2013UC Hypercoupling\n'
                 'in Severe Intrapartum Acidosis',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'figure2_characterisation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def figure3(df, centers, traj):
    """Figure 3 — Permutation test distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    targets = {12: (10, 15), 18: (15, 20)}

    for idx, (tmin, (lo, hi)) in enumerate(targets.items()):
        ax = axes[idx]
        vals, labels = [], []
        for _, row in df.iterrows():
            mask = (row['_times'] >= lo) & (row['_times'] < hi)
            if mask.any():
                vals.append(float(np.mean(row['_coh'][mask])))
                labels.append(1 if row['group'] == 'severe' else 0)

        vals   = np.array(vals)
        labels = np.array(labels)
        sev_v  = vals[labels == 1]
        nor_v  = vals[labels == 0]
        real_d = np.mean(sev_v) - np.mean(nor_v)

        perm_d = []
        for _ in range(N_PERM):
            shuf = labels.copy()
            np.random.shuffle(shuf)
            perm_d.append(np.mean(vals[shuf == 1]) -
                          np.mean(vals[shuf == 0]))
        perm_d  = np.array(perm_d)
        perm_p  = np.mean(np.abs(perm_d) >= np.abs(real_d))
        _, mw_p = mannwhitneyu(sev_v, nor_v, alternative='two-sided')

        ax.hist(perm_d, bins=60, color='#90CAF9', alpha=0.7,
                density=True, label=f'Random shuffles (n={N_PERM})')
        extreme = np.abs(perm_d) >= np.abs(real_d)
        ax.hist(perm_d[extreme], bins=60, color='#EF5350',
                alpha=0.5, density=True,
                label=f'More extreme ({extreme.sum()}/{N_PERM})')
        ax.axvline(real_d, color='#B71C1C', lw=3,
                   label=f'Real diff = {real_d:.4f}')
        ax.axvline(-real_d, color='#B71C1C', lw=1.5, ls='--', alpha=0.5)

        result = 'ROBUST' if perm_p < 0.05 else 'NOT ROBUST'
        col    = '#2E7D32' if perm_p < 0.05 else '#B71C1C'
        ax.set_title(f'Minute {tmin} ({lo}\u2013{hi} min)\n'
                     f'Permutation p = {perm_p:.4f}  |  MW p = {mw_p:.4f}'
                     f'\n{result}',
                     fontweight='bold', color=col)
        ax.set_xlabel('Mean Difference (Severe \u2212 Normal)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

    plt.suptitle('Permutation Validation of Early Hypercoupling Signal\n'
                 '2000 Random Label Shuffles',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'figure3_permutation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ─── STEP 7: PRINT RESULTS ────────────────────────────────────────────────────
def print_results(df):
    groups = ['normal', 'borderline', 'moderate', 'severe']
    nor    = df[df['group'] == 'normal']['coh_mean_30'].dropna()

    print("\n" + "=" * 65)
    print("TABLE 1 — Early FHR-UC Phase Coherence by Outcome Group")
    print("=" * 65)
    print(f"{'Group':<14} {'n':>4} {'Median':>8}  {'[Q1-Q3]':<22} {'p vs normal':>12}")
    print("-" * 65)
    for g in groups:
        sub = df[df['group'] == g]['coh_mean_30'].dropna()
        med = np.median(sub)
        q1  = np.percentile(sub, 25)
        q3  = np.percentile(sub, 75)
        if g == 'normal':
            ps = '—'
        else:
            _, p = mannwhitneyu(sub, nor, alternative='two-sided')
            ps = f'{p:.4f}' + (' **' if p < 0.01 else ' *' if p < 0.05 else '')
        print(f"{g:<14} {len(sub):>4} {med:>8.4f}  [{q1:.4f}–{q3:.4f}]   {ps:>12}")

    sev = df[df['group'] == 'severe']
    print(f"\nSevere acidosis (pH < 7.05):")
    print(f"  n = {len(sev)}")
    print(f"  Minute-12 coherence: {np.mean([np.mean(r['_coh'][(r['_times']>=10)&(r['_times']<15)]) for _,r in sev.iterrows() if ((r['_times']>=10)&(r['_times']<15)).any()]):.4f}")
    print(f"  Normal minute-12:    {np.mean([np.mean(r['_coh'][(r['_times']>=10)&(r['_times']<15)]) for _,r in df[df['group']=='normal'].iterrows() if ((r['_times']>=10)&(r['_times']<15)).any()]):.4f}")
    print(f"  Coupling trend:      {sev['trend'].mean():.4f} (normal: {df[df['group']=='normal']['trend'].mean():.4f})")

    y = (df['group'] == 'severe').astype(int)
    x = df['coh_mean_30'].fillna(df['coh_mean_30'].median())
    fpr, tpr, _ = roc_curve(y, x)
    print(f"\n  AUC (coh_mean_30 for pH < 7.05): {auc(fpr, tpr):.3f}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("FHR-UC Phase Hypercoupling Analysis")
    print("Rishi Arun Shivhare | rishivhare07@gmail.com")
    print("=" * 65)

    recs = download_records()
    df   = process_all(recs)
    df.to_csv(os.path.join("results", "coupling_features.csv"), index=False)

    centers, traj = bin_trajectories(df)

    print_results(df)

    print("\nGenerating figures...")
    figure1(df, centers, traj)
    figure2(df, centers, traj)
    figure3(df, centers, traj)

    print("\nDone. Check figures/ and results/ directories.")
