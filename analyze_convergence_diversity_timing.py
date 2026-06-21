#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_convergence_diversity_timing.py — KBS hazirligi icin uc analizi tek seferde.

A) CONVERGENCE CURVES
    Her algoritma icin iteration basina best-so-far fitness ortalamasi (30 run).
    Hakem sorusu: "How fast does each algorithm converge?"

B) DIVERSITY (Exploration/Exploitation)
    Her algoritma icin iteration basina populasyondaki 9D vektor std'i (normalize).
    Hakem sorusu: "How does each algorithm balance exploration/exploitation?"

C) TIMING TABLOSU
    Her algoritma icin ortalama wall-time + std + speedup.
    Hakem sorusu: "Is GWO's win worth the computational cost?"

Veri kaynagi: gwo_runs/run_*/gwo_running.db (her algoritma icin 30 run)

Kullanim:
    python analyze_convergence_diversity_timing.py
    # -> kbs_analyses/
    #    convergence.png
    #    diversity.png
    #    timing_table.csv
    #    timing_table.tex
    #    summary.json
"""

import os
import sys
import json
import pickle
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Akademik figur ayarlari
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

ALGOS = ["gwo", "pso", "ga", "abc", "ho", "toa", "raindrop", "fno"]
# Renkler (Q1 paper standart paleti)
COLORS = {
    "gwo":      "#1f77b4",
    "pso":      "#ff7f0e",
    "ga":       "#2ca02c",
    "abc":      "#d62728",
    "ho":       "#9467bd",
    "toa":      "#8c564b",
    "raindrop": "#e377c2",
    "fno":      "#7f7f7f",
}
LABELS = {
    "gwo": "GWO", "pso": "PSO", "ga": "GA", "abc": "ABC",
    "ho": "HO", "toa": "TOA", "raindrop": "Raindrop", "fno": "FNO",
}
N_RUNS = 30
OUT_DIR = Path("kbs_analyses")
OUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# 1. CONVERGENCE
# =============================================================================

def load_convergence(algo: str) -> tuple:
    """30 run icin iteration basina best-so-far fitness matrisi.

    Returns:
        conv: shape (n_runs_found, max_iter+1)
        n_iter: int (init dahil)
    """
    runs_dir = Path(f"{algo}_runs")
    if not runs_dir.exists():
        print(f"  [!] {runs_dir} yok, atlandi.")
        return None, 0

    all_curves = []
    max_iter = 0
    for run_id in range(1, N_RUNS + 1):
        db_path = runs_dir / f"run_{run_id:03d}" / f"{algo}_running.db"
        if not db_path.exists():
            continue
        try:
            con = sqlite3.connect(str(db_path))
            cur = con.cursor()
            # iteration_stats.global_best_at_end (best-so-far)
            rows = cur.execute(
                "SELECT iteration, global_best_at_end FROM iteration_stats "
                "ORDER BY iteration ASC"
            ).fetchall()
            con.close()
            if not rows:
                continue
            iters, bests = zip(*rows)
            all_curves.append(list(bests))
            max_iter = max(max_iter, len(bests))
        except Exception as e:
            print(f"  [!] {db_path}: {e}")
            continue

    if not all_curves:
        return None, 0

    # Tek matrise pad (kisa olanlari son deger ile tamamla)
    mat = np.full((len(all_curves), max_iter), np.nan)
    for i, c in enumerate(all_curves):
        mat[i, :len(c)] = c
        if len(c) < max_iter:
            mat[i, len(c):] = c[-1]  # son best ile doldur

    return mat, max_iter


def plot_convergence(conv_by_algo: dict) -> Path:
    """8 algoritma icin convergence figuru (mean + std band)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo in ALGOS:
        if algo not in conv_by_algo:
            continue
        mat = conv_by_algo[algo]
        if mat is None:
            continue
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat, axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, color=COLORS[algo], lw=1.8,
                label=f"{LABELS[algo]} (n={mat.shape[0]})")
        ax.fill_between(x, mean - std, mean + std,
                        color=COLORS[algo], alpha=0.10)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best-so-far Validation RMSE (normalized)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.95)
    ax.set_title("Convergence behavior of 8 metaheuristic algorithms "
                 "(mean over 30 runs, shaded = ±1 SD)")
    plt.tight_layout()
    out_path = OUT_DIR / "convergence.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


# =============================================================================
# 2. DIVERSITY (DECISION SPACE)
# =============================================================================

def load_diversity(algo: str) -> tuple:
    """Iteration basina decision-space std'i (9D normalize edilmis).

    Mantik: Her iteration'da pop'taki vektorler -> her boyut icin std -> mean over boyutlar.
    Boyutlar farkli olcekte oldugu icin once min-max normalize edilir.
    """
    runs_dir = Path(f"{algo}_runs")
    if not runs_dir.exists():
        return None, 0

    diversity_curves = []
    max_iter = 0
    for run_id in range(1, N_RUNS + 1):
        db_path = runs_dir / f"run_{run_id:03d}" / f"{algo}_running.db"
        if not db_path.exists():
            continue
        try:
            con = sqlite3.connect(str(db_path))
            cur = con.cursor()
            # Iteration basina tum trial'larin vector'leri (BLOB pickle)
            rows = cur.execute(
                "SELECT iteration, vector FROM trial_history "
                "WHERE vector IS NOT NULL ORDER BY iteration ASC"
            ).fetchall()
            con.close()
            if not rows:
                continue

            # iteration -> [vectors]
            from collections import defaultdict
            iter_vecs = defaultdict(list)
            for it, blob in rows:
                try:
                    v = pickle.loads(blob)
                    if isinstance(v, np.ndarray) and v.ndim == 1:
                        iter_vecs[it].append(v)
                except Exception:
                    continue

            if not iter_vecs:
                continue

            # Tum vektorleri toplayip global min-max bul
            all_vecs = np.array([v for vs in iter_vecs.values() for v in vs])
            if all_vecs.shape[0] < 2:
                continue
            vmin = all_vecs.min(axis=0)
            vmax = all_vecs.max(axis=0)
            vrange = np.where((vmax - vmin) > 1e-12, vmax - vmin, 1.0)

            curve = []
            for it in sorted(iter_vecs.keys()):
                vs = np.array(iter_vecs[it])
                if vs.shape[0] < 2:
                    curve.append(np.nan)
                    continue
                normalized = (vs - vmin) / vrange
                # Her boyut icin std, sonra ortalamala
                diversity = float(np.mean(np.std(normalized, axis=0)))
                curve.append(diversity)

            diversity_curves.append(curve)
            max_iter = max(max_iter, len(curve))
        except Exception as e:
            print(f"  [!] {db_path} diversity err: {e}")
            continue

    if not diversity_curves:
        return None, 0

    mat = np.full((len(diversity_curves), max_iter), np.nan)
    for i, c in enumerate(diversity_curves):
        mat[i, :len(c)] = c

    return mat, max_iter


def plot_diversity(div_by_algo: dict) -> Path:
    """8 algoritma icin diversity (decision-space) figuru."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo in ALGOS:
        if algo not in div_by_algo:
            continue
        mat = div_by_algo[algo]
        if mat is None:
            continue
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat, axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, color=COLORS[algo], lw=1.8,
                label=f"{LABELS[algo]}")
        ax.fill_between(x, mean - std, mean + std,
                        color=COLORS[algo], alpha=0.08)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Population diversity (mean σ over 9 normalized dimensions)")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.95)
    ax.set_title("Exploration → exploitation transition\n"
                 "(higher diversity = exploration; lower = exploitation; 30-run mean)")
    plt.tight_layout()
    out_path = OUT_DIR / "diversity.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


# =============================================================================
# 3. TIMING TABLOSU
# =============================================================================

def collect_timing(algo: str) -> dict:
    """30 run icin toplam wall-time istatistikleri."""
    runs_dir = Path(f"{algo}_runs")
    if not runs_dir.exists():
        return None

    durations = []           # her run'in toplam suresi (s)
    iter_durations = []      # her iteration'in suresi (s) (tum runlar pooled)
    for run_id in range(1, N_RUNS + 1):
        db_path = runs_dir / f"run_{run_id:03d}" / f"{algo}_running.db"
        if not db_path.exists():
            continue
        try:
            con = sqlite3.connect(str(db_path))
            cur = con.cursor()
            # Toplam: iteration_stats.duration_seconds sum
            rows = cur.execute(
                "SELECT duration_seconds FROM iteration_stats"
            ).fetchall()
            con.close()
            if not rows:
                continue
            run_durations = [r[0] for r in rows if r[0] is not None]
            if not run_durations:
                continue
            total = float(sum(run_durations))
            durations.append(total)
            iter_durations.extend(run_durations)
        except Exception as e:
            print(f"  [!] {db_path} timing err: {e}")
            continue

    if not durations:
        return None

    durations = np.array(durations)
    iter_durations = np.array(iter_durations)

    return {
        "algorithm": LABELS[algo],
        "n_runs": int(len(durations)),
        "mean_total_s": float(durations.mean()),
        "std_total_s":  float(durations.std()),
        "median_total_s": float(np.median(durations)),
        "min_total_s":  float(durations.min()),
        "max_total_s":  float(durations.max()),
        "mean_iter_s":  float(iter_durations.mean()),
        "std_iter_s":   float(iter_durations.std()),
        "n_iters_total": int(len(iter_durations)),
    }


def write_timing_table(timings: dict) -> tuple:
    """CSV + LaTeX tablo yaz."""
    # Ortalamaya gore sirala (artan = en hizli ilk)
    rows = sorted(timings.values(), key=lambda x: x["mean_total_s"])
    baseline_fastest = rows[0]["mean_total_s"]

    # CSV
    csv_path = OUT_DIR / "timing_table.csv"
    with open(csv_path, "w") as f:
        f.write("algorithm,n_runs,mean_total_min,std_total_min,"
                "mean_iter_s,std_iter_s,speedup_vs_fastest\n")
        for r in rows:
            speedup = r["mean_total_s"] / baseline_fastest
            f.write(f"{r['algorithm']},{r['n_runs']},"
                    f"{r['mean_total_s']/60:.2f},{r['std_total_s']/60:.2f},"
                    f"{r['mean_iter_s']:.1f},{r['std_iter_s']:.1f},"
                    f"{speedup:.2f}\n")

    # LaTeX
    tex_path = OUT_DIR / "timing_table.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Computational cost of 8 metaheuristic algorithms "
                "(30 runs each, 40 individuals $\\times$ 50 iterations = "
                "2000 surrogate evaluations per run).}\n")
        f.write("\\label{tab:timing}\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Algorithm & Mean (min) & SD (min) & Mean/iter (s) & "
                "Cost ratio$^a$ \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            speedup = r["mean_total_s"] / baseline_fastest
            f.write(f"{r['algorithm']} & {r['mean_total_s']/60:.1f} & "
                    f"{r['std_total_s']/60:.1f} & "
                    f"{r['mean_iter_s']:.1f} & "
                    f"{speedup:.2f}$\\times$ \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\n$^a$ Cost ratio = mean wall-time relative to fastest "
                f"algorithm ({rows[0]['algorithm']}).\n")
        f.write("\\end{table}\n")

    return csv_path, tex_path


# =============================================================================
# 4. ANA AKIS
# =============================================================================

def main():
    print("=" * 70)
    print("KBS HAZIRLIK ANALIZLERI: convergence + diversity + timing")
    print("=" * 70)

    # CONVERGENCE
    print("\n[1/3] Convergence verisi okunuyor...")
    conv_by_algo = {}
    for algo in ALGOS:
        print(f"  {algo}...", end=" ", flush=True)
        mat, n_iter = load_convergence(algo)
        if mat is None:
            print("YOK")
            continue
        conv_by_algo[algo] = mat
        print(f"OK (n_runs={mat.shape[0]}, n_iter={n_iter})")

    if conv_by_algo:
        conv_path = plot_convergence(conv_by_algo)
        print(f"  Figur: {conv_path}")

    # DIVERSITY
    print("\n[2/3] Diversity verisi okunuyor...")
    div_by_algo = {}
    for algo in ALGOS:
        print(f"  {algo}...", end=" ", flush=True)
        mat, n_iter = load_diversity(algo)
        if mat is None:
            print("YOK")
            continue
        div_by_algo[algo] = mat
        print(f"OK (n_runs={mat.shape[0]}, n_iter={n_iter})")

    if div_by_algo:
        div_path = plot_diversity(div_by_algo)
        print(f"  Figur: {div_path}")

    # TIMING
    print("\n[3/3] Timing analizi...")
    timings = {}
    for algo in ALGOS:
        t = collect_timing(algo)
        if t is None:
            print(f"  {algo}: YOK")
            continue
        timings[algo] = t
        print(f"  {algo}: {t['mean_total_s']/60:.1f} +/- {t['std_total_s']/60:.1f} min "
              f"(n={t['n_runs']})")

    if timings:
        csv_path, tex_path = write_timing_table(timings)
        print(f"  CSV:   {csv_path}")
        print(f"  LaTeX: {tex_path}")

    # JSON ozet
    summary = {
        "convergence": {
            algo: {
                "n_runs": int(mat.shape[0]),
                "final_mean_loss": float(np.nanmean(mat[:, -1])),
                "final_std_loss":  float(np.nanstd(mat[:, -1])),
            }
            for algo, mat in conv_by_algo.items()
        },
        "diversity": {
            algo: {
                "n_runs": int(mat.shape[0]),
                "initial_mean": float(np.nanmean(mat[:, 0])),
                "final_mean":   float(np.nanmean(mat[:, -1])),
                "decay_ratio":  float(np.nanmean(mat[:, -1]) / max(np.nanmean(mat[:, 0]), 1e-9)),
            }
            for algo, mat in div_by_algo.items()
        },
        "timing": timings,
        "created_at": datetime.now().isoformat(),
    }
    out_json = OUT_DIR / "summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Ozet: {out_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
