#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_algorithms.py - 8 algoritmanin agregat karsilastirmasi.

Faz 1 tamamlandiktan sonra (her algoritmanin 30 kosusu bitince) tum
algoritmalarin sonuclarini birlestirip:
- Boxplot, convergence overlay, bar chart, radar
- Wilcoxon signed-rank pairwise (Holm correction)
- Friedman test (multi-method)
- LaTeX tablolar
- Aggregate JSON

Sart: final_training_v4.py --skip-final her algoritma icin onceden calistirilmis
olmali. Yani her <algo>_final_results/analysis_summary.json mevcut.

Kullanim:
    python compare_algorithms.py --algos abc,ga,pso,ho,fno,raindrop,toa
    python compare_algorithms.py --algos ho,fno,raindrop,pso  # sadece 4 algo

Cikti: comparison_results/
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("COMPARE")

# Akademik makale formati
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix',
})


# =============================================================================
# 1. VERI YUKLEME
# =============================================================================

def load_algo_summary(algo: str, base_dir: str = ".") -> dict:
    """Bir algoritmanin analysis_summary.json'unu yukle."""
    path = os.path.join(base_dir, f"{algo}_final_results", "analysis_summary.json")
    if not os.path.exists(path):
        logger.warning(f"Bulunamadi: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_all_summaries(algos: list, base_dir: str = ".") -> dict:
    """Tum algoritmalarin summary'lerini yukle.

    Returns:
        {algo: summary_dict} - bulunamayanlar atlanir
    """
    summaries = {}
    for algo in algos:
        s = load_algo_summary(algo, base_dir)
        if s is not None:
            summaries[algo] = s
            logger.info(f"  {algo}: {len(s.get('all_losses', []))} kosu yuklendi")
        else:
            logger.warning(f"  {algo}: ATLANDI (data yok)")
    return summaries


# =============================================================================
# 2. ISTATISTIKSEL TESTLER
# =============================================================================

def wilcoxon_pairwise(summaries: dict) -> dict:
    """Tum algoritma ciftleri icin Wilcoxon signed-rank test.

    Same seeds across all algos -> paired test mumkun.
    Holm-Bonferroni correction uygulanir.
    """
    from scipy.stats import wilcoxon

    algos = list(summaries.keys())
    pairs = list(combinations(algos, 2))
    results = []

    for a1, a2 in pairs:
        losses1 = summaries[a1].get('all_losses', [])
        losses2 = summaries[a2].get('all_losses', [])

        # Esit boyut sart (paired test icin)
        n = min(len(losses1), len(losses2))
        if n < 5:
            results.append({
                'a1': a1, 'a2': a2,
                'n': n, 'p_value': None, 'statistic': None,
                'note': 'insufficient samples'
            })
            continue

        l1 = np.array(losses1[:n])
        l2 = np.array(losses2[:n])
        try:
            stat, pval = wilcoxon(l1, l2, alternative='two-sided',
                                   zero_method='zsplit')
            results.append({
                'a1': a1, 'a2': a2, 'n': n,
                'statistic': float(stat),
                'p_value': float(pval),
                'mean_diff': float(np.mean(l1) - np.mean(l2)),
            })
        except Exception as e:
            results.append({
                'a1': a1, 'a2': a2, 'n': n,
                'p_value': None, 'note': str(e)
            })

    # Holm-Bonferroni correction
    valid = [r for r in results if r.get('p_value') is not None]
    valid.sort(key=lambda r: r['p_value'])
    m = len(valid)
    for i, r in enumerate(valid):
        adj_p = min(1.0, r['p_value'] * (m - i))
        r['p_adjusted'] = adj_p
        r['significant'] = adj_p < 0.05

    return {'pairs': results, 'method': 'Wilcoxon signed-rank + Holm correction'}


def friedman_test(summaries: dict) -> dict:
    """Friedman test: multi-method comparison (>=3 algos gerekir)."""
    from scipy.stats import friedmanchisquare

    if len(summaries) < 3:
        return {'p_value': None,
                'note': f'Friedman requires >=3 algos, got {len(summaries)}'}

    # Tum algoritmalarin ayni boyutta loss vektoru olmali
    n = min(len(s.get('all_losses', [])) for s in summaries.values())
    if n < 5:
        return {'p_value': None, 'note': 'insufficient samples'}

    data = [np.array(s['all_losses'][:n]) for s in summaries.values()]
    stat, pval = friedmanchisquare(*data)

    # Ranks (her seed icin algoritma siralamasi)
    matrix = np.array(data)  # shape: (n_algos, n_seeds)
    ranks = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        ranks[:, j] = np.argsort(np.argsort(matrix[:, j])) + 1
    mean_ranks = ranks.mean(axis=1)

    return {
        'statistic': float(stat),
        'p_value': float(pval),
        'n_seeds': n,
        'n_algos': len(summaries),
        'mean_ranks': {algo: float(r) for algo, r in zip(summaries.keys(), mean_ranks)},
        'best_by_rank': list(summaries.keys())[int(np.argmin(mean_ranks))],
    }


def cohens_d(x: list, y: list) -> float:
    """Cohen's d effect size."""
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled = np.sqrt(((nx-1)*x.var(ddof=1) + (ny-1)*y.var(ddof=1)) / (nx+ny-2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


# =============================================================================
# 3. GRAFIKLER
# =============================================================================

def plot_boxplot_comparison(summaries: dict, save_path: str):
    """Tum algoritmalar yan yana boxplot."""
    algos = list(summaries.keys())
    data = [summaries[a].get('all_losses', []) for a in algos]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, labels=[a.upper() for a in algos],
                     showmeans=True, meanline=True, patch_artist=True)

    # Renk paleti
    colors = plt.cm.tab10(np.linspace(0, 1, len(algos)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel('Best Loss (RMSE)')
    ax.set_xlabel('Algorithm')
    ax.set_title(f'Algorithm Comparison: Best Loss Across {len(summaries)} Algorithms (n=30 runs each)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Boxplot: {save_path}")


def plot_stats_bar(summaries: dict, save_path: str):
    """Algoritmalar arasinda mean ± std bar chart."""
    algos = list(summaries.keys())
    means = [summaries[a]['statistics']['mean'] for a in algos]
    stds = [summaries[a]['statistics']['std'] for a in algos]
    mins = [summaries[a]['statistics']['min'] for a in algos]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(algos))
    width = 0.35

    bars1 = ax.bar(x - width/2, means, width, yerr=stds, capsize=4,
                    label='Mean ± Std', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, mins, width, label='Min', color='green', alpha=0.7)

    # En iyiyi vurgula (yesil bar uzerinde)
    best_idx = int(np.argmin(mins))
    bars2[best_idx].set_edgecolor('darkgreen')
    bars2[best_idx].set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel('Loss (RMSE)')
    ax.set_title('Algorithm Performance Summary')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Stats bar: {save_path}")


def plot_friedman_ranks(friedman: dict, save_path: str):
    """Friedman test mean ranks grafigi."""
    if 'mean_ranks' not in friedman:
        return
    ranks = friedman['mean_ranks']
    sorted_items = sorted(ranks.items(), key=lambda x: x[1])
    algos = [a.upper() for a, _ in sorted_items]
    rank_vals = [r for _, r in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(algos, rank_vals, color='coral', alpha=0.7)
    bars[0].set_color('seagreen')  # en iyi
    bars[0].set_alpha(0.9)

    ax.set_xlabel('Mean Rank (lower is better)')
    ax.set_title(
        f"Friedman Test: Mean Ranks "
        f"(stat={friedman['statistic']:.2f}, p={friedman['p_value']:.2e})"
    )
    ax.grid(True, alpha=0.3, axis='x')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Friedman ranks: {save_path}")


def plot_wilcoxon_heatmap(wilcoxon_res: dict, summaries: dict, save_path: str):
    """Wilcoxon adjusted p-value heatmap (algo × algo)."""
    algos = list(summaries.keys())
    n = len(algos)
    matrix = np.ones((n, n))

    for r in wilcoxon_res['pairs']:
        if r.get('p_adjusted') is None:
            continue
        i = algos.index(r['a1'])
        j = algos.index(r['a2'])
        matrix[i, j] = r['p_adjusted']
        matrix[j, i] = r['p_adjusted']

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([a.upper() for a in algos], rotation=45, ha='right')
    ax.set_yticklabels([a.upper() for a in algos])
    plt.colorbar(im, ax=ax, label='Adjusted p-value (Holm)')

    # p-value'lari hucrelere yaz
    for i in range(n):
        for j in range(n):
            if i != j:
                txt = f"{matrix[i,j]:.3f}"
                if matrix[i,j] < 0.05:
                    txt += "*"
                ax.text(j, i, txt, ha='center', va='center',
                        color='white' if matrix[i,j] < 0.05 else 'black',
                        fontsize=8)
            else:
                ax.text(j, i, "-", ha='center', va='center', fontsize=8)

    ax.set_title('Wilcoxon Signed-Rank Pairwise Comparison\n(Holm-corrected p-values; * = significant at α=0.05)')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Wilcoxon heatmap: {save_path}")


# =============================================================================
# 4. LATEX TABLOLAR
# =============================================================================

def latex_main_comparison(summaries: dict, friedman: dict, save_path: str):
    """Makale ana karsilastirma tablosu (8 algoritma × 5 metric + rank)."""
    algos = list(summaries.keys())
    ranks = friedman.get('mean_ranks', {})

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Algorithm Performance Comparison Across 30 Independent Runs}",
        r"\label{tab:algo_comparison}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Algorithm} & \textbf{Min} & \textbf{Mean} & \textbf{Std} & "
        r"\textbf{Median} & \textbf{CV (\%)} & \textbf{Rank} \\",
        r"\hline",
    ]

    # Mean ranks'a gore sirala
    if ranks:
        algos_sorted = sorted(algos, key=lambda a: ranks.get(a, 999))
    else:
        algos_sorted = sorted(algos, key=lambda a: summaries[a]['statistics']['min'])

    for algo in algos_sorted:
        st = summaries[algo]['statistics']
        rank = ranks.get(algo, '-')
        rank_str = f"{rank:.2f}" if isinstance(rank, float) else str(rank)
        lines.append(
            f"{algo.upper()} & {st['min']:.6f} & {st['mean']:.6f} & "
            f"{st['std']:.6f} & {st['median']:.6f} & {st['cv_percent']:.2f} & "
            f"{rank_str} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"LaTeX main: {save_path}")


def latex_wilcoxon_table(wilcoxon_res: dict, save_path: str):
    """Wilcoxon pairwise sonuclarini LaTeX tablosu olarak yaz."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Wilcoxon Signed-Rank Pairwise Test (Holm-corrected p-values)}",
        r"\label{tab:wilcoxon}",
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"\textbf{Algo 1} & \textbf{Algo 2} & \textbf{Mean Diff} & "
        r"\textbf{p (raw)} & \textbf{p (Holm)} \\",
        r"\hline",
    ]
    # Sirala p_adjusted'a gore
    pairs = sorted(
        [p for p in wilcoxon_res['pairs'] if p.get('p_value') is not None],
        key=lambda r: r.get('p_adjusted', 1.0)
    )
    for r in pairs:
        sig = "*" if r.get('significant') else ""
        lines.append(
            f"{r['a1'].upper()} & {r['a2'].upper()} & {r.get('mean_diff', 0):.6f} & "
            f"{r['p_value']:.4f} & {r.get('p_adjusted', 1.0):.4f}{sig} \\\\"
        )
    lines.extend([
        r"\hline",
        r"\multicolumn{5}{l}{\small * significant at $\alpha=0.05$ (Holm-corrected)} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"LaTeX Wilcoxon: {save_path}")


# =============================================================================
# 5. ANA AKIS
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="8 algoritma agregat karsilastirma (Faz 1 sonrasi)",
    )
    parser.add_argument('--algos', type=str,
                        default='abc,ga,pso,ho,fno,raindrop,toa',
                        help='Algoritmalar (virgulle ayrilmis)')
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Algoritmalarin <algo>_final_results/ klasorlerinin bulundugu yer')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                        help='Cikti klasoru')
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(',')]
    output_dir = args.output_dir
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # 1. Veri yukle
    logger.info(f"[1/5] {len(algos)} algoritma yukleniyor...")
    summaries = load_all_summaries(algos, args.base_dir)
    if len(summaries) < 2:
        logger.error("En az 2 algoritma sonucu gerekli!")
        sys.exit(1)
    logger.info(f"{len(summaries)} algoritma basariyla yuklendi.")

    # 2. Friedman test
    logger.info("\n[2/5] Friedman test...")
    friedman = friedman_test(summaries)
    if friedman.get('p_value') is not None:
        logger.info(f"  Friedman: stat={friedman['statistic']:.4f}, p={friedman['p_value']:.4e}")
        logger.info(f"  Best by rank: {friedman['best_by_rank']}")
    else:
        logger.warning(f"  Friedman test atlandi: {friedman.get('note')}")

    # 3. Wilcoxon pairwise
    logger.info("\n[3/5] Wilcoxon pairwise + Holm correction...")
    wilcoxon_res = wilcoxon_pairwise(summaries)
    sig_pairs = [p for p in wilcoxon_res['pairs'] if p.get('significant')]
    logger.info(f"  {len(sig_pairs)}/{len(wilcoxon_res['pairs'])} cift anlamli (p_adj<0.05)")

    # 4. Grafikler
    logger.info("\n[4/5] Grafikler uretiliyor...")
    plot_boxplot_comparison(summaries, os.path.join(figures_dir, "01_boxplot_all_algos.png"))
    plot_stats_bar(summaries, os.path.join(figures_dir, "02_stats_bar.png"))
    plot_friedman_ranks(friedman, os.path.join(figures_dir, "03_friedman_ranks.png"))
    plot_wilcoxon_heatmap(wilcoxon_res, summaries,
                          os.path.join(figures_dir, "04_wilcoxon_heatmap.png"))

    # 5. LaTeX tablolar + JSON ozet
    logger.info("\n[5/5] Tablolar ve JSON ozet...")
    latex_main_comparison(summaries, friedman,
                           os.path.join(tables_dir, "comparison_main.tex"))
    latex_wilcoxon_table(wilcoxon_res,
                          os.path.join(tables_dir, "wilcoxon_pairwise.tex"))

    aggregate = {
        'algorithms': list(summaries.keys()),
        'n_algos': len(summaries),
        'friedman_test': friedman,
        'wilcoxon_pairwise': wilcoxon_res,
        'per_algo_stats': {a: summaries[a]['statistics'] for a in summaries},
        'per_algo_best_loss': {a: summaries[a]['statistics']['min'] for a in summaries},
        'created_at': datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "aggregate_summary.json"), 'w') as f:
        json.dump(aggregate, f, indent=2)

    # Ekran ozeti
    print()
    print("=" * 70)
    print(f"KARSILASTIRMA OZETI ({len(summaries)} algoritma)")
    print("=" * 70)
    print(f"{'Algoritma':<12} {'Min':<12} {'Mean':<12} {'Std':<12} {'Rank':<6}")
    print("-" * 70)
    ranks = friedman.get('mean_ranks', {})
    for algo in sorted(summaries.keys(), key=lambda a: ranks.get(a, 999)):
        st = summaries[algo]['statistics']
        rank = ranks.get(algo, '-')
        rank_str = f"{rank:.2f}" if isinstance(rank, float) else str(rank)
        print(f"{algo.upper():<12} {st['min']:<12.6f} {st['mean']:<12.6f} {st['std']:<12.6f} {rank_str:<6}")
    print("=" * 70)
    print(f"\nCikti: {output_dir}/")
    print(f"  figures/  - 4 grafik (boxplot, bar, friedman, wilcoxon heatmap)")
    print(f"  tables/   - 2 LaTeX tablo")
    print(f"  aggregate_summary.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
