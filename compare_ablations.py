#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_ablations.py — Faz 3 Ablation Analizi.

Bilimsel cerceve:
    Baseline (full) algoritma ile 3 ablation senaryosu karSilaStirilir:
      - novmd     : VMD kapali (K=1, ham sinyal)
      - noatt     : TemporalAttention kapali (saf TCN)
      - nopenalty : Process-Aware Divergence Penalty kapali (klasik RMSE)

    Her senaryo icin 30 baGimsiz koSu ve ayni seed seti (joint ile esit
    buce). Final-training loss (test set RMSE) birincil metriktir;
    final results yoksa optimizasyon-ici en iyi loss'a fallback yapilir.

Istatistik:
    - 1-vs-1: full vs her ablation (Mann-Whitney U, one-sided)
                Effect size: Cohen's d + Cliff's delta
    - K-way:  Friedman testi (4 senaryo, n=30 koSu)
                Post-hoc: Nemenyi-style mean ranks

Kullanim:
    # Tek komutla 4 ciktidan tablo + grafik uretir
    python compare_ablations.py --algo ga \\
        --ablations novmd,noatt,nopenalty \\
        --output-dir ablation_results

    # Sadece DB-fallback (final results henuz olmadi):
    python compare_ablations.py --algo ga --use-db

Cikti:
    ablation_results/
    ├── ablation_summary.json      # Tum istatistikler
    ├── ablation_table.tex         # LaTeX tablo (paper 5.3 icin)
    ├── ablation_table.md          # Markdown tablo (paper preview)
    └── figures/
        ├── ablation_boxplot.png   # 4 senaryo boxplot
        ├── ablation_delta_pct.png # % degisim bar chart
        └── ablation_ranks.png     # Friedman mean ranks
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ABL] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ABL")

# Akademik grafik formati
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

# Ablation gosterim isimleri (paper icin)
ABLATION_LABELS = {
    None:           ('Full',          '#2c7bb6'),  # baseline (mavi)
    'nodwt':        ('No DWT',        '#d7191c'),  # kirmizi
    'nobidir':      ('No Bidir.',     '#fdae61'),  # turuncu
    'waveletfixed': ('Wavelet sym5',  '#abd9e9'),  # acik mavi
}


# =============================================================================
# 1. SONUC TOPLAMA (final_results oncelikli, DB fallback)
# =============================================================================

def _read_db_best_loss(db_path: str) -> float:
    """Tek bir DB'den optimizasyon-ici en iyi loss'u oku."""
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute("""
            SELECT MIN(loss) FROM trial_history
            WHERE loss IS NOT NULL AND loss < 999999
        """).fetchone()
        conn.close()
        if row and row[0] is not None:
            return float(row[0])
    except Exception as e:
        logger.debug(f"DB okuma hatasi {db_path}: {e}")
    return None


def collect_losses(algo: str, ablation: str = None,
                   n_runs: int = 30, force_db: bool = False) -> dict:
    """Bir senaryo icin 30 koSunun loss listesini topla.

    Args:
        algo: 'ga', 'pso', vb.
        ablation: None (baseline), 'novmd', 'noatt', 'nopenalty'
        n_runs: beklenen koSu sayisi
        force_db: True ise final_results'i atla, doGrudan DB'den oku

    Returns:
        {
            'scenario': 'full' veya 'novmd' vb.,
            'algo': str,
            'ablation': str veya None,
            'losses': [float, ...],
            'source': 'final_results' veya 'db_opt_loss',
            'n_collected': int,
            'n_expected': int,
        }
    """
    suffix = f"_{ablation}" if ablation else ""
    scenario = ablation if ablation else 'full'

    # YENI dizin sablonu (Faz 4 ablation_runner.py uyumlu):
    #   - ablation_runs/{algo}_{ablation}_analysis.json    (final_results yerine)
    #   - ablation_runs/{algo}_{ablation}/run_001..030/    (XX_runs yerine)
    # ESKI dizin sablonu (Faz 1 final_training_v4.py uyumlu):
    #   - {algo}_final_results/analysis_summary.json
    #   - {algo}_runs/run_001..030/
    # Once ESKi (full/baseline icin) sonra YENi (ablation icin) deneriz.
    candidate_summaries = []
    candidate_runs_dirs = []
    if ablation:
        # Yeni sablon
        candidate_summaries.append(f"ablation_runs/{algo}{suffix}_analysis.json")
        candidate_runs_dirs.append(f"ablation_runs/{algo}{suffix}")
        # Geriye uyum (eski)
        candidate_summaries.append(f"{algo}{suffix}_final_results/analysis_summary.json")
        candidate_runs_dirs.append(f"{algo}{suffix}_runs")
    else:
        # Baseline (full) - Faz 1 sonucu
        candidate_summaries.append(f"{algo}_final_results/analysis_summary.json")
        candidate_runs_dirs.append(f"{algo}_runs")

    summary_path = None
    final_dir = None
    for cs in candidate_summaries:
        if os.path.exists(cs):
            summary_path = cs
            final_dir = os.path.dirname(cs) or "."
            break

    runs_dir = None
    for cr in candidate_runs_dirs:
        if os.path.isdir(cr):
            runs_dir = cr
            break

    result = {
        'scenario': scenario,
        'algo': algo,
        'ablation': ablation,
        'losses': [],
        'source': None,
        'n_collected': 0,
        'n_expected': n_runs,
    }

    # ONCELIK 1: final_results/analysis_summary.json
    if not force_db and summary_path is not None:
        try:
            with open(summary_path, "rb") as f:
                raw = f.read().replace(b"\x00", b"").strip()
            text = raw.decode("utf-8", errors="ignore")
            # multi-JSON konkatenasyonuna karsi: ilk obj
            depth = 0
            end = 0
            for i, c in enumerate(text):
                if c == '{': depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            summary = json.loads(text[:end])
            all_losses = summary.get('all_losses', [])
            if all_losses:
                result['losses'] = [float(x) for x in all_losses[:n_runs]]
                result['n_collected'] = len(result['losses'])
                result['source'] = 'final_results'
                logger.info(
                    f"  [{scenario:9s}] {result['n_collected']}/{n_runs} kosu "
                    f"(kaynak: {summary_path} -- final-train)"
                )
                return result
        except Exception as e:
            logger.warning(f"  [{scenario}] summary okunamadi: {e}")

    # FALLBACK: DB'lerden optimizasyon-ici best
    if runs_dir is None or not os.path.isdir(runs_dir):
        logger.warning(
            f"  [{scenario}] HIC veri yok: aranan summary={candidate_summaries}, "
            f"runs={candidate_runs_dirs}"
        )
        return result

    losses = []
    for i in range(1, n_runs + 1):
        run_dir = os.path.join(runs_dir, f"run_{i:03d}")
        # iki olasi DB ismi
        candidates = [
            os.path.join(run_dir, f"{algo}_running.db"),
            os.path.join(run_dir, f"{algo}{suffix}_running.db"),
        ]
        db_path = next((p for p in candidates if os.path.exists(p)), None)
        if db_path is None:
            continue
        loss = _read_db_best_loss(db_path)
        if loss is not None:
            losses.append(loss)

    result['losses'] = losses
    result['n_collected'] = len(losses)
    result['source'] = 'db_opt_loss'
    logger.info(
        f"  [{scenario:9s}] {len(losses)}/{n_runs} kosu "
        f"(kaynak: {runs_dir}/*/db -- opt-best, FALLBACK)"
    )
    return result


# =============================================================================
# 2. ISTATISTIK (1-vs-1 ve K-way)
# =============================================================================

def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1))
                     / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    greater = sum(1 for xi in x for yi in y if xi > yi)
    less    = sum(1 for xi in x for yi in y if xi < yi)
    return float((greater - less) / (nx * ny))


def _interpret_effect(cohens_d, cliffs_delta) -> str:
    if cohens_d is None or cliffs_delta is None:
        return "N/A"
    abs_d = abs(cohens_d); abs_dlt = abs(cliffs_delta)
    if abs_d >= 0.8 or abs_dlt >= 0.474:
        return "large"
    if abs_d >= 0.5 or abs_dlt >= 0.33:
        return "medium"
    if abs_d >= 0.2 or abs_dlt >= 0.147:
        return "small"
    return "negligible"


def compare_pair(baseline: dict, ablation: dict) -> dict:
    """Full vs ablation tek-kanalli istatistiksel kar Silastirma.

    Hipotez: H1: loss_ablation > loss_baseline (ablation daha kotu yapar)
    Mann-Whitney U one-sided.
    """
    from scipy import stats as sp_stats

    b = np.array(baseline['losses'])
    a = np.array(ablation['losses'])

    if len(b) < 5 or len(a) < 5:
        return {
            'scenario': ablation['scenario'],
            'n_baseline': len(b),
            'n_ablation': len(a),
            'baseline_mean': float(b.mean()) if len(b) else None,
            'ablation_mean': float(a.mean()) if len(a) else None,
            'delta_pct': None,
            'mannwhitney_p': None,
            'significant': None,
            'cohens_d': None,
            'cliffs_delta': None,
            'effect': 'N/A',
            'warning': 'En az 5 koSu gerekli',
        }

    # one-sided Mann-Whitney: H1: ablation > baseline
    try:
        _, p_val = sp_stats.mannwhitneyu(a, b, alternative='greater')
    except Exception:
        p_val = None

    cohens_d = _cohens_d(a, b)        # a-b: ablation daha kotu mu?
    cliffs_d = _cliffs_delta(a, b)
    effect   = _interpret_effect(cohens_d, cliffs_d)

    pct_change = ((a.mean() - b.mean()) / b.mean() * 100) if b.mean() != 0 else None

    return {
        'scenario': ablation['scenario'],
        'n_baseline': len(b),
        'n_ablation': len(a),
        'baseline_mean': float(b.mean()),
        'baseline_std':  float(b.std(ddof=1)),
        'ablation_mean': float(a.mean()),
        'ablation_std':  float(a.std(ddof=1)),
        'delta_pct':     float(pct_change) if pct_change is not None else None,
        'mannwhitney_p': float(p_val) if p_val is not None else None,
        'significant':   bool(p_val < 0.05) if p_val is not None else None,
        'cohens_d':      cohens_d,
        'cliffs_delta':  cliffs_d,
        'effect':        effect,
    }


def friedman_kway(scenarios: list) -> dict:
    """4 senaryo (full + 3 ablation) icin Friedman testi + mean ranks."""
    from scipy import stats as sp_stats

    # Eşit boyut zorunlu - en küçüğüne kırp
    min_n = min(len(s['losses']) for s in scenarios if s['losses'])
    if min_n < 5:
        return {'p_value': None, 'statistic': None, 'mean_ranks': None,
                'warning': 'n<5'}

    data = np.array([s['losses'][:min_n] for s in scenarios])  # (k, n)
    # Friedman input: her satir farkli "tedavi", her sutun ayni "blok"
    try:
        stat, p_val = sp_stats.friedmanchisquare(*data)
    except Exception:
        return {'p_value': None, 'statistic': None, 'mean_ranks': None,
                'warning': 'friedman calismadi'}

    # Mean ranks: her sutun (koSu) icin tedavileri rank et, sonra ortala
    # Düşük loss -> düşük rank (1 = en iyi)
    ranks = np.zeros_like(data)
    for j in range(data.shape[1]):
        ranks[:, j] = sp_stats.rankdata(data[:, j])
    mean_ranks = ranks.mean(axis=1).tolist()

    return {
        'p_value': float(p_val),
        'statistic': float(stat),
        'n_used': int(min_n),
        'k_scenarios': len(scenarios),
        'mean_ranks': {s['scenario']: float(r)
                       for s, r in zip(scenarios, mean_ranks)},
    }


# =============================================================================
# 3. GORSELLESTIRME
# =============================================================================

def plot_boxplot(scenarios: list, output_path: str, algo: str) -> None:
    """Tum senaryolarin boxplot karSilastirmasi.

    Y ekseni: HP search asamasinda her kosunun en iyi dogrulama RMSE'sidir
    (analysis_summary.json 'all_losses' alani). Final training loss DEGIL.
    """
    # Times New Roman fallback'i (Linux'ta Liberation Serif metric-uyumlu)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif",
                       "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    data = [s['losses'] for s in scenarios]
    labels = [ABLATION_LABELS[s['ablation']][0] for s in scenarios]
    colors = [ABLATION_LABELS[s['ablation']][1] for s in scenarios]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)

    # DOGRU etiket: HP search dagilimi (final training loss DEGIL)
    ax.set_ylabel('Validation RMSE (HP Search)')
    ax.set_xlabel('Scenario')
    n_runs = min(len(s["losses"]) for s in scenarios)
    ax.set_title(f'{algo.upper()} - Ablation Comparison '
                 f'(n = {n_runs} runs per scenario)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.savefig(output_path)
    plt.close()
    logger.info(f"  Grafik: {output_path}")


def plot_delta_bars(comparisons: list, output_path: str, algo: str) -> None:
    """Ablation senaryolarinin baseline'a gore %farkini gosteren bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    names  = [ABLATION_LABELS[c['scenario'] if c['scenario'] != 'full' else None][0]
              for c in comparisons]
    deltas = [c['delta_pct'] if c['delta_pct'] is not None else 0
              for c in comparisons]
    colors = [ABLATION_LABELS[c['scenario'] if c['scenario'] != 'full' else None][1]
              for c in comparisons]

    bars = ax.bar(names, deltas, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_ylabel(r'$\Delta$ vs Full Baseline (%)')
    ax.set_title(f'{algo.upper()} - Ablation Senaryolarinin Baseline\'a Gore '
                 f'RMSE Yuzde Degisimi')
    ax.grid(True, alpha=0.3, axis='y')

    # Bar uzerine deGer + anlamlilik *
    for bar, c in zip(bars, comparisons):
        h = bar.get_height()
        sig = ' *' if c.get('significant') else ''
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f'{h:+.1f}%{sig}', ha='center',
                va='bottom' if h >= 0 else 'top', fontsize=9)

    plt.savefig(output_path)
    plt.close()
    logger.info(f"  Grafik: {output_path}")


def plot_mean_ranks(friedman_result: dict, output_path: str, algo: str) -> None:
    """Friedman mean ranks bar chart."""
    if not friedman_result.get('mean_ranks'):
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ranks_dict = friedman_result['mean_ranks']
    names  = [ABLATION_LABELS[k if k != 'full' else None][0] for k in ranks_dict.keys()]
    values = list(ranks_dict.values())
    colors = [ABLATION_LABELS[k if k != 'full' else None][1] for k in ranks_dict.keys()]

    bars = ax.bar(names, values, color=colors, edgecolor='black', alpha=0.85)
    ax.set_ylabel('Mean Rank (lower = better)')

    p_val = friedman_result.get('p_value')
    p_str = f'p < 0.0001' if p_val and p_val < 1e-4 else f'p = {p_val:.4f}' if p_val else 'N/A'
    ax.set_title(f'{algo.upper()} Friedman Test - Mean Ranks\n'
                 f'$\\chi^2$ = {friedman_result.get("statistic", 0):.2f}, {p_str}, '
                 f'n = {friedman_result.get("n_used", 0)}')

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"  Grafik: {output_path}")


# =============================================================================
# 4. LATEX + MARKDOWN TABLO
# =============================================================================

def generate_latex_table(baseline: dict, comparisons: list,
                          friedman: dict, output_path: str, algo: str) -> None:
    """Paper 5.3 icin LaTeX tablo."""
    b_mean = float(np.mean(baseline['losses']))
    b_std  = float(np.std(baseline['losses'], ddof=1))

    rows = [
        f"    {algo.upper()} (Full)         & {b_mean:.6f} & {b_std:.6f} "
        f"& -- & -- & -- & -- \\\\",
    ]
    for c in comparisons:
        sig = '*' if c.get('significant') else ''
        p_val = c.get('mannwhitney_p')
        p_str = f"$<10^{{-4}}${sig}" if p_val is not None and p_val < 1e-4 else \
                f"{p_val:.4f}{sig}" if p_val is not None else '--'
        delta = f"{c['delta_pct']:+.2f}" if c.get('delta_pct') is not None else '--'
        d_val = f"{c['cohens_d']:.2f}" if c.get('cohens_d') is not None else '--'
        dlt   = f"{c['cliffs_delta']:.2f}" if c.get('cliffs_delta') is not None else '--'
        label = ABLATION_LABELS[c['scenario']][0]
        rows.append(
            f"    {algo.upper()} ({label})    & {c['ablation_mean']:.6f} "
            f"& {c['ablation_std']:.6f} & {delta}\\% & {p_str} & {d_val} & {dlt} \\\\"
        )

    fried_str = ""
    if friedman.get('p_value') is not None:
        chi = friedman['statistic']; p = friedman['p_value']
        n = friedman['n_used']
        p_disp = "<10^{-4}" if p < 1e-4 else f"{p:.4f}"
        fried_str = (f"\nFriedman test ($n={n}$): $\\chi^2 = {chi:.2f}$, "
                     f"$p = {p_disp}$. ")

    latex = r"""\begin{table}[htbp]
\centering
\caption{Ablation Study (""" + algo.upper() + r""", 30 independent runs each).
$\Delta$: percentage change in mean RMSE relative to full baseline (positive = worse).
Mann--Whitney U test, one-sided ($H_1$: $\mathrm{RMSE}_{\text{ablation}} > \mathrm{RMSE}_{\text{full}}$).
* $p < 0.05$. Cohen's $d$ and Cliff's $\delta$ measure effect size (large $\geq 0.8 / 0.474$).""" \
+ fried_str + r"""}
\label{tab:ablation_""" + algo + r"""}
\begin{tabular}{lccrrcc}
\hline
\textbf{Scenario} & \textbf{Mean RMSE} & \textbf{Std} & \textbf{$\Delta$ (\%)} & \textbf{$p$-value} & \textbf{Cohen's $d$} & \textbf{Cliff's $\delta$} \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}
"""
    with open(output_path, 'w') as f:
        f.write(latex)
    logger.info(f"  LaTeX tablo: {output_path}")


def generate_md_table(baseline: dict, comparisons: list,
                       friedman: dict, output_path: str, algo: str) -> None:
    """Paper preview icin Markdown tablo."""
    b_mean = float(np.mean(baseline['losses']))
    b_std  = float(np.std(baseline['losses'], ddof=1))

    lines = [
        f"# Ablation Karsilastirmasi - {algo.upper()}",
        "",
        f"| Senaryo | Mean RMSE | Std | $\\Delta$ % | Mann-Whitney $p$ | Cohen's $d$ | Cliff's $\\delta$ | Effect |",
        f"|---|---|---|---|---|---|---|---|",
        f"| **{algo.upper()} Full** | {b_mean:.6f} | {b_std:.6f} | -- | -- | -- | -- | baseline |",
    ]
    for c in comparisons:
        sig = ' \\*' if c.get('significant') else ''
        p_val = c.get('mannwhitney_p')
        p_str = f"<0.0001{sig}" if p_val is not None and p_val < 1e-4 else \
                f"{p_val:.4f}{sig}" if p_val is not None else '--'
        label = ABLATION_LABELS[c['scenario']][0]
        delta = f"{c['delta_pct']:+.2f}" if c.get('delta_pct') is not None else '--'
        d_val = f"{c['cohens_d']:.2f}" if c.get('cohens_d') is not None else '--'
        dlt   = f"{c['cliffs_delta']:.2f}" if c.get('cliffs_delta') is not None else '--'
        lines.append(
            f"| {algo.upper()} {label} | {c['ablation_mean']:.6f} "
            f"| {c['ablation_std']:.6f} | {delta} | {p_str} | {d_val} | {dlt} | {c.get('effect', 'N/A')} |"
        )

    if friedman.get('p_value') is not None:
        lines += [
            "",
            f"**Friedman test** (k={friedman['k_scenarios']} senaryo, n={friedman['n_used']} koSu): ",
            f"$\\chi^2 = {friedman['statistic']:.2f}$, $p = {friedman['p_value']:.2e}$",
            "",
            "**Mean ranks (lower = better):**",
        ]
        for scen, rank in friedman['mean_ranks'].items():
            label = ABLATION_LABELS[scen if scen != 'full' else None][0]
            lines.append(f"- {label}: {rank:.2f}")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"  Markdown tablo: {output_path}")


# =============================================================================
# 5. ANA AKIS
# =============================================================================

def analyze(algo: str, ablations: list, output_dir: str,
            n_runs: int, force_db: bool) -> dict:
    """Tum ablation analizi tek elden."""
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"FAZ 3 ABLATION ANALIZI - {algo.upper()}")
    logger.info(f"Ablation senaryolari: {ablations}")
    logger.info(f"Veri kaynagi tercih: {'DB (force)' if force_db else 'final_results -> DB fallback'}")
    logger.info("=" * 60)

    # 1) Baseline (full)
    logger.info("\n[1/4] Sonuclar toplaniyor...")
    baseline = collect_losses(algo, ablation=None, n_runs=n_runs, force_db=force_db)
    if not baseline['losses']:
        logger.error(f"Baseline ({algo} full) icin veri yok. Cikiyor.")
        return {}

    # 2) Ablation senaryolari
    ablation_results = []
    for abl in ablations:
        res = collect_losses(algo, ablation=abl, n_runs=n_runs, force_db=force_db)
        if res['losses']:
            ablation_results.append(res)

    if not ablation_results:
        logger.error("Hicbir ablation senaryosu icin veri bulunamadi. Cikiyor.")
        return {}

    # 3) 1-vs-1 karSilastirma
    logger.info("\n[2/4] 1-vs-1 istatistik testleri...")
    comparisons = []
    for abl_res in ablation_results:
        cmp_result = compare_pair(baseline, abl_res)
        comparisons.append(cmp_result)
        if cmp_result.get('delta_pct') is not None:
            sig = '*' if cmp_result['significant'] else 'ns'
            logger.info(
                f"  {ABLATION_LABELS[abl_res['ablation']][0]:14s} vs Full: "
                f"Δ={cmp_result['delta_pct']:+.2f}% | "
                f"p={cmp_result['mannwhitney_p']:.2e} ({sig}) | "
                f"d={cmp_result['cohens_d']:.2f} | "
                f"δ={cmp_result['cliffs_delta']:.2f} ({cmp_result['effect']})"
            )

    # 4) K-way Friedman
    logger.info("\n[3/4] Friedman K-way testi...")
    all_scenarios = [baseline] + ablation_results
    friedman = friedman_kway(all_scenarios)
    if friedman.get('p_value') is not None:
        logger.info(
            f"  Friedman chi^2 = {friedman['statistic']:.2f}, "
            f"p = {friedman['p_value']:.2e} (n={friedman['n_used']})"
        )
        for scen, rank in friedman['mean_ranks'].items():
            label = ABLATION_LABELS[scen if scen != 'full' else None][0]
            logger.info(f"    {label:14s} mean rank = {rank:.2f}")

    # 5) Grafikler
    logger.info("\n[4/4] Grafikler ve tablolar uretiliyor...")
    plot_boxplot(all_scenarios,
                 os.path.join(fig_dir, 'ablation_boxplot.png'), algo)
    plot_delta_bars(comparisons,
                    os.path.join(fig_dir, 'ablation_delta_pct.png'), algo)
    plot_mean_ranks(friedman,
                    os.path.join(fig_dir, 'ablation_ranks.png'), algo)

    # 6) Tablolar
    generate_latex_table(baseline, comparisons, friedman,
                         os.path.join(output_dir, 'ablation_table.tex'), algo)
    generate_md_table(baseline, comparisons, friedman,
                       os.path.join(output_dir, 'ablation_table.md'), algo)

    # 7) JSON ozet
    summary = {
        'algo': algo,
        'ablations': ablations,
        'n_runs_expected': n_runs,
        'created_at': datetime.now().isoformat(),
        'baseline': {
            'scenario': 'full',
            'n_collected': baseline['n_collected'],
            'source': baseline['source'],
            'mean':   float(np.mean(baseline['losses'])),
            'std':    float(np.std(baseline['losses'], ddof=1)),
            'median': float(np.median(baseline['losses'])),
            'min':    float(np.min(baseline['losses'])),
            'max':    float(np.max(baseline['losses'])),
            'losses': baseline['losses'],
        },
        'ablation_results': [
            {
                'scenario': a['scenario'],
                'n_collected': a['n_collected'],
                'source': a['source'],
                'mean':   float(np.mean(a['losses'])),
                'std':    float(np.std(a['losses'], ddof=1)),
                'median': float(np.median(a['losses'])),
                'min':    float(np.min(a['losses'])),
                'max':    float(np.max(a['losses'])),
                'losses': a['losses'],
            }
            for a in ablation_results
        ],
        'pairwise_comparisons': comparisons,
        'friedman': friedman,
    }
    summary_path = os.path.join(output_dir, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  JSON ozet: {summary_path}")

    logger.info("\n" + "=" * 60)
    logger.info("ABLATION ANALIZI TAMAMLANDI")
    logger.info(f"Cikti: {output_dir}/")
    logger.info("=" * 60)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Faz 3 Ablation Analizi (full vs novmd/noatt/nopenalty)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--algo', type=str, default='ga',
                        help="Baseline algoritma (default: ga)")
    parser.add_argument('--ablations', type=str, default='nodwt,nobidir,waveletfixed',
                        help="Test edilecek ablation senaryolari, virgulle (default: nodwt,nobidir,waveletfixed)")
    parser.add_argument('--n-runs', type=int, default=30,
                        help="Beklenen kosu sayisi (default: 30)")
    parser.add_argument('--output-dir', type=str, default='ablation_results',
                        help="Cikti dizini (default: ablation_results)")
    parser.add_argument('--use-db', action='store_true',
                        help="Final results'i atla, dogrudan DB'den oku "
                             "(final eGitim henuz olmadi ise)")

    args = parser.parse_args()
    ablations = [a.strip().lower() for a in args.ablations.split(',') if a.strip()]

    # Geçerli ablation'lar (v9 — PADP kaldirildi)
    valid = {'nodwt', 'nobidir', 'waveletfixed'}
    invalid = set(ablations) - valid
    if invalid:
        parser.error(f"Bilinmeyen ablation: {invalid}. Gecerli: {valid}")

    analyze(args.algo, ablations, args.output_dir,
            args.n_runs, force_db=args.use_db)
    return 0


if __name__ == '__main__':
    sys.exit(main())
