#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
final_training_v4.py - Asama 3: algoritma-agnostic final egitim + analiz.

Eski final_training_and_analysis.py (ABC-specific) ve _ga.py (GA-specific)
yerine gecer. Tek scriptle herhangi bir algoritma icin Asama 3 calistirilir.

Akis:
    1. <algo>_runs/run_NNN/<algo>_running.db dosyalarini tara (30 kosu)
    2. Her kosudan en iyi parametreleri cikart
    3. Istatistikler (mean, std, min, max, percentiles)
    4. Analiz figurleri (convergence, boxplot, parameter distribution)
    5. En iyi kosunun parametreleri ile GPU final egitim (--skip-final ile atla)
    6. JSON ozet + LaTeX tablo + figurler -> <algo>_final_results/

Kullanim:
    # ABC icin Asama 3
    python final_training_v4.py --algo abc

    # PSO icin, custom dizinler ile
    python final_training_v4.py --algo pso --runs-dir pso_runs --n-runs 30

    # Sadece analiz, final egitim atla (GPU yoksa)
    python final_training_v4.py --algo abc --skip-final

    # Sadece final egitim (analiz tamamlandiysa)
    python final_training_v4.py --algo abc --only-final
"""

import os
# Thread limit (TRUBA SLURM uyumlulugu)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_NUM_THREADS", "1")

import sys
import json
import sqlite3
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FINAL_V4")

# =============================================================================
# MATPLOTLIB - Akademik makale formati
# =============================================================================
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
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'mathtext.fontset': 'stix',
})


# =============================================================================
# 1. SONUC TOPLAMA (30 DB -> aggregate)
# =============================================================================

def collect_all_results(algo: str, runs_dir: str, num_runs: int = 30) -> tuple:
    """Tum kosulardan en iyi sonuclari topla.

    Returns:
        (all_results, convergence_data)
        all_results: list of dict {run_id, best_loss, params}
        convergence_data: dict {run_id: {iterations, best_losses}}
    """
    all_results = []
    convergence_data = {}

    for i in range(1, num_runs + 1):
        run_dir = os.path.join(runs_dir, f"run_{i:03d}")
        db_path = os.path.join(run_dir, f"{algo}_running.db")

        if not os.path.exists(db_path):
            logger.warning(f"Run {i:03d}: DB yok ({db_path})")
            continue

        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

            # Eski ABC semas? (cycle) mi, yeni v4 semas? (iteration) mi?
            col_info = cur.execute("PRAGMA table_info(trial_history)").fetchall()
            col_names = [c[1] for c in col_info]
            iter_col = 'cycle' if 'cycle' in col_names else 'iteration'

            # En iyi sonuc (DWT+BiLSTM semasi, v9)
            cur.execute("""
                SELECT loss, dwt_wavelet, dwt_level, dwt_mode,
                       look_back, hidden, layers,
                       dropout, lr, batch
                FROM trial_history
                WHERE loss IS NOT NULL AND loss < 999999
                ORDER BY loss ASC LIMIT 1
            """)
            row = cur.fetchone()
            if not row or row[0] is None:
                conn.close()
                continue

            best_loss = row[0]
            best_params = {
                'dwt_wavelet': row[1], 'dwt_level': row[2], 'dwt_mode': row[3],
                'look_back': row[4], 'hidden': row[5], 'layers': row[6],
                'dropout': row[7], 'lr': row[8], 'batch': row[9],
            }

            # Convergence (running minimum) — iter_col otomatik algilandi
            cur.execute(f"""
                SELECT {iter_col}, MIN(loss)
                FROM trial_history
                WHERE loss IS NOT NULL AND loss < 999999
                GROUP BY {iter_col}
                ORDER BY {iter_col}
            """)
            rows = cur.fetchall()
            if rows:
                iters = [r[0] for r in rows]
                losses = [r[1] for r in rows]
                running_min = []
                cur_min = float('inf')
                for L in losses:
                    cur_min = min(cur_min, L)
                    running_min.append(cur_min)
                convergence_data[i] = {
                    'iterations': iters,
                    'best_losses': running_min,
                }

            all_results.append({
                'run_id': i,
                'best_loss': best_loss,
                'params': best_params,
            })
            conn.close()

        except Exception as e:
            logger.error(f"Run {i:03d} okuma hatasi: {e}")

    logger.info(f"{algo}: {len(all_results)}/{num_runs} kosu basariyla okundu")
    return all_results, convergence_data


# =============================================================================
# 2. ISTATISTIK
# =============================================================================

def compute_statistics(all_results: list) -> dict:
    """Tum kosulardaki en iyi loss'larin istatistiklerini hesapla."""
    losses = [r['best_loss'] for r in all_results]
    if not losses:
        return {}

    return {
        'n_runs': len(losses),
        'min': float(np.min(losses)),
        'max': float(np.max(losses)),
        'mean': float(np.mean(losses)),
        'median': float(np.median(losses)),
        'std': float(np.std(losses)),
        'q1': float(np.percentile(losses, 25)),
        'q3': float(np.percentile(losses, 75)),
        'cv_percent': float(np.std(losses) / np.mean(losses) * 100),
    }


def print_statistical_summary(algo: str, stats: dict, all_results: list):
    """Ekrana istatistik tablosu yazdir."""
    print()
    print("=" * 60)
    print(f"{algo.upper()} ISTATISTIKSEL OZET (n={stats.get('n_runs', 0)})")
    print("=" * 60)
    print(f"  Minimum:   {stats.get('min', 0):.6f}")
    print(f"  Maximum:   {stats.get('max', 0):.6f}")
    print(f"  Mean:      {stats.get('mean', 0):.6f}")
    print(f"  Median:    {stats.get('median', 0):.6f}")
    print(f"  Std Dev:   {stats.get('std', 0):.6f}")
    print(f"  Q1 (25%):  {stats.get('q1', 0):.6f}")
    print(f"  Q3 (75%):  {stats.get('q3', 0):.6f}")
    print(f"  CV (%):    {stats.get('cv_percent', 0):.2f}")
    print("=" * 60)

    # En iyi 5
    sorted_results = sorted(all_results, key=lambda r: r['best_loss'])
    print("En iyi 5 kosu:")
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. Run {r['run_id']:03d}: loss={r['best_loss']:.6f}")
    print()


# =============================================================================
# 3. GRAFIKLER
# =============================================================================

def plot_convergence(algo: str, conv_data: dict, save_path: str):
    """30 kosunun yakinsama egrilerini cikar (ortalama + std band)."""
    if not conv_data:
        return

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(conv_data)))

    # Bireysel egriler (silik)
    for idx, (run_id, data) in enumerate(conv_data.items()):
        plt.plot(data['iterations'], data['best_losses'],
                 alpha=0.3, linewidth=0.7, color=colors[idx])

    # Ortalama egri (kalin)
    all_iters = sorted(set(it for d in conv_data.values() for it in d['iterations']))
    mean_curve = []
    std_curve = []
    for it in all_iters:
        vals = []
        for d in conv_data.values():
            if it in d['iterations']:
                idx = d['iterations'].index(it)
                vals.append(d['best_losses'][idx])
        if vals:
            mean_curve.append(np.mean(vals))
            std_curve.append(np.std(vals))

    plt.plot(all_iters, mean_curve, 'r-', linewidth=2.5, label=f'{algo.upper()} mean')
    plt.fill_between(all_iters,
                      np.array(mean_curve) - np.array(std_curve),
                      np.array(mean_curve) + np.array(std_curve),
                      color='red', alpha=0.2, label='±1 Std')

    plt.xlabel('Iteration')
    plt.ylabel('Best Loss (RMSE)')
    plt.title(f'{algo.upper()} Convergence Curves (n={len(conv_data)} runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Convergence plot: {save_path}")


def plot_boxplot(algo: str, all_results: list, save_path: str):
    """30 kosunun box plot + histogram."""
    losses = [r['best_loss'] for r in all_results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].boxplot(losses, vert=True)
    axes[0].set_ylabel('Best Loss (RMSE)')
    axes[0].set_title(f'{algo.upper()} Best Loss Distribution')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(losses, bins=15, edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(losses), color='r', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(losses):.6f}')
    axes[1].axvline(np.min(losses), color='g', linestyle='--', linewidth=2,
                    label=f'Min: {np.min(losses):.6f}')
    axes[1].set_xlabel('Best Loss (RMSE)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{algo.upper()} Loss Histogram (n={len(losses)})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Boxplot: {save_path}")


def plot_parameter_distribution(algo: str, all_results: list, save_path: str):
    """30 kosunun parametre dagilimlari."""
    params_values = {}
    for r in all_results:
        for k, v in r['params'].items():
            params_values.setdefault(k, []).append(v)

    n_params = len(params_values)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, (pname, pvals) in enumerate(params_values.items()):
        ax = axes[idx]
        # Kategorik (string) parametreler icin bar chart, numerik icin histogram
        is_categorical = any(isinstance(v, str) for v in pvals)
        if is_categorical:
            from collections import Counter
            counts = Counter(str(v) for v in pvals)
            # Sirali key listesi (frequency desc)
            labels = sorted(counts.keys(), key=lambda k: -counts[k])
            values = [counts[lbl] for lbl in labels]
            bars = ax.bar(range(len(labels)), values,
                          edgecolor='black', alpha=0.75)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            # Mode (en sik kategori) etiketi
            mode_lbl = labels[0]
            ax.axhline(np.mean(values), color='r', linestyle='--', linewidth=1.0,
                       label=f'Mode: {mode_lbl} ({counts[mode_lbl]}×)')
        else:
            try:
                pvals_num = np.asarray(pvals, dtype=float)
                ax.hist(pvals_num, bins=10, edgecolor='black', alpha=0.7)
                ax.axvline(float(np.mean(pvals_num)), color='r',
                           linestyle='--', linewidth=1.5,
                           label=f'Mean: {np.mean(pvals_num):.4g}')
            except (TypeError, ValueError):
                # Yine de cevirilemiyor: kategorik gibi davran
                from collections import Counter
                counts = Counter(str(v) for v in pvals)
                labels = sorted(counts.keys(), key=lambda k: -counts[k])
                values = [counts[lbl] for lbl in labels]
                ax.bar(range(len(labels)), values, edgecolor='black', alpha=0.75)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_xlabel(pname)
        ax.set_ylabel('Freq')
        ax.set_title(f'{pname} dist')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Bos subplot'lari gizle
    for idx in range(len(params_values), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'{algo.upper()} Best Parameters Distribution (n={len(all_results)} runs)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Parameter distribution: {save_path}")


# =============================================================================
# 4. LATEX TABLO
# =============================================================================

def generate_latex_table(algo: str, stats: dict, best_run: dict, save_path: str):
    """Makale icin LaTeX tablosu uret."""
    bp = best_run.get('params', {})
    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{algo.upper()} Optimization Results Across 30 Independent Runs}}
\\label{{tab:{algo}_results}}
\\begin{{tabular}}{{lc}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\hline
Number of Runs & {stats.get('n_runs', 0)} \\\\
Minimum Loss   & {stats.get('min', 0):.6f} \\\\
Maximum Loss   & {stats.get('max', 0):.6f} \\\\
Mean Loss      & {stats.get('mean', 0):.6f} \\\\
Median Loss    & {stats.get('median', 0):.6f} \\\\
Std. Deviation & {stats.get('std', 0):.6f} \\\\
CV (\\%)       & {stats.get('cv_percent', 0):.2f} \\\\
\\hline
\\multicolumn{{2}}{{l}}{{\\textbf{{Best Run Hyperparameters:}}}} \\\\
dwt\\_wavelet   & {bp.get('dwt_wavelet', '-')} \\\\
dwt\\_level     & {bp.get('dwt_level', '-')} \\\\
dwt\\_mode      & {bp.get('dwt_mode', '-')} \\\\
look\\_back     & {bp.get('look_back', '-')} \\\\
hidden         & {bp.get('hidden', '-')} \\\\
layers         & {bp.get('layers', '-')} \\\\
dropout        & {bp.get('dropout', 0):.4f} \\\\
lr             & {bp.get('lr', 0):.5f} \\\\
batch          & {bp.get('batch', '-')} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open(save_path, 'w') as f:
        f.write(latex)
    logger.info(f"LaTeX table: {save_path}")


# =============================================================================
# 5. FINAL EGITIM (GPU)
# =============================================================================

def run_final_training(algo: str, best_params: dict, models_dir: str, figures_dir: str,
                       use_dwt: bool = True, use_bidirectional: bool = True):
    """En iyi paramlarla tam veri GPU final egitim.

    objective.py'daki WindForecastObjective is_final_training=True modunu kullanir.
    Bu mod tum verinin tamamini, 100 epoch, GPU ile egitim yapar.
    """
    logger.info("=" * 60)
    logger.info(f"FINAL EGITIM BASLIYOR (algo: {algo})")
    logger.info("=" * 60)
    logger.info(f"Best params: {best_params}")

    # Lazy import - GPU + heavy deps
    import config
    import objective as objective_module
    from objective import WindForecastObjective
    from utils import HP

    # Models ve figures dizinlerini ayarla.
    # KRITIK: objective.py "from config import MODELS_DIR_PATH" ile ice aktarir,
    # yani config'i degistirmek yetmez — objective modulunun kendi kopyasi da
    # guncellenmeli. Aksi halde tum algoritmalar proje kokündeki models/ klasorune
    # yazar ve birbirinin modelini ezer.
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    config.MODELS_DIR_PATH = models_dir
    config.FIGURES_DIR_PATH = figures_dir
    objective_module.MODELS_DIR_PATH = models_dir   # <-- asil fix
    objective_module.FIGURES_DIR_PATH = figures_dir  # <-- asil fix
    logger.info(f"Model kayit dizini: {models_dir}")
    logger.info(f"Figur kayit dizini: {figures_dir}")

    # HP enum'a cevir (objective dict[HP] bekliyor)
    hp_params = {
        HP.DWT_WAVELET: best_params['dwt_wavelet'],
        HP.DWT_LEVEL:   best_params['dwt_level'],
        HP.DWT_MODE:    best_params['dwt_mode'],
        HP.LOOK_BACK:   best_params['look_back'],
        HP.HIDDEN:      best_params['hidden'],
        HP.LAYERS:      best_params['layers'],
        HP.DROPOUT:     best_params['dropout'],
        HP.LR:          best_params['lr'],
        HP.BATCH:       best_params['batch'],
    }

    # Final egitim — GPU, tum veri, 100 epoch
    obj = WindForecastObjective(
        mode="global",
        is_final_training=True,  # KRITIK: full data + 100 epoch
        show=1,
        use_gpu=True,
        use_dwt=use_dwt,
        use_bidirectional=use_bidirectional,
    )
    logger.info(f"Ablation flags: use_dwt={use_dwt}, use_bidirectional={use_bidirectional}")

    final_loss = obj(hp_params, log_prefix=f"[FINAL_{algo.upper()}]")
    logger.info(f"Final loss: {final_loss:.6f}")
    return final_loss


# =============================================================================
# 6. ANA AKIS
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Asama 3: algoritma-agnostic final egitim + analiz (v4)",
    )
    parser.add_argument('--algo', type=str, required=True,
                        help='Algoritma adi (abc/ga/pso/ho/fno/raindrop)')
    parser.add_argument('--runs-dir', type=str, default=None,
                        help='Kosu dizini (varsayilan: <algo>_runs)')
    parser.add_argument('--n-runs', type=int, default=30,
                        help='Beklenen kosu sayisi (varsayilan: 30)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Cikti dizini (varsayilan: <algo>_final_results)')
    parser.add_argument('--skip-final', action='store_true',
                        help='Final GPU egitimini atla (sadece analiz)')
    parser.add_argument('--only-final', action='store_true',
                        help='Sadece final egitim (analiz atlanmaz, JSON\'dan okur)')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['nodwt', 'nobidir', 'waveletfixed'],
                        help='Faz 4 ablation: optimizasyon hangi flag\'le yapildiysa '
                             'final egitim de ayni flag\'le yapilmali. '
                             'nodwt: DWT kapali, nobidir: BiLSTM->LSTM, '
                             'waveletfixed: wavelet sym5 sabit.')
    args = parser.parse_args()

    # Ablation flag'lerini dönüstür
    abl = args.ablation
    use_dwt = (abl != 'nodwt')
    use_bidirectional = (abl != 'nobidir')
    # waveletfixed env var ile parametre mapping etkilenecek; flag uretmiyoruz

    algo = args.algo.lower()
    abl_suffix = f"_{abl}" if abl else ""
    runs_dir = args.runs_dir or f"{algo}{abl_suffix}_runs"
    output_dir = args.output_dir or f"{algo}{abl_suffix}_final_results"
    figures_dir = os.path.join(output_dir, "figures")
    models_dir = os.path.join(output_dir, "models")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "analysis_summary.json")

    # =====================================================================
    # ASAMA 3a: Analiz (--only-final degilse)
    # =====================================================================
    if not args.only_final:
        logger.info(f"[1/4] Sonuclar toplaniyor ({runs_dir}/run_001..{args.n_runs:03d})")
        all_results, conv_data = collect_all_results(algo, runs_dir, args.n_runs)
        if not all_results:
            logger.error("Hic sonuc bulunamadi!")
            sys.exit(1)

        # Istatistik
        stats = compute_statistics(all_results)
        print_statistical_summary(algo, stats, all_results)

        # En iyi kosu
        best_run = min(all_results, key=lambda r: r['best_loss'])
        logger.info(f"\nEn iyi kosu: Run {best_run['run_id']:03d}")
        logger.info(f"Best loss: {best_run['best_loss']:.6f}")
        logger.info(f"Best params: {best_run['params']}")

        # Figurler
        logger.info("\n[2/4] Figurler uretiliyor...")
        plot_convergence(algo, conv_data,
                          os.path.join(figures_dir, f"{algo}_convergence.png"))
        plot_boxplot(algo, all_results,
                      os.path.join(figures_dir, f"{algo}_boxplot.png"))
        plot_parameter_distribution(algo, all_results,
                                      os.path.join(figures_dir, f"{algo}_params_dist.png"))

        # LaTeX tablo
        logger.info("\n[3/4] LaTeX tablo uretiliyor...")
        generate_latex_table(algo, stats, best_run,
                              os.path.join(output_dir, f"{algo}_table.tex"))

        # JSON ozet
        summary = {
            'algorithm': algo,
            'n_runs_found': len(all_results),
            'n_runs_expected': args.n_runs,
            'best_run_id': best_run['run_id'],
            'best_loss': best_run['best_loss'],
            'best_params': best_run['params'],
            'statistics': stats,
            'all_losses': [r['best_loss'] for r in all_results],
            'all_run_ids': [r['run_id'] for r in all_results],
            'created_at': datetime.now().isoformat(),
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Ozet kaydedildi: {summary_path}")
    else:
        logger.info(f"[--only-final] Analiz atlandi, mevcut JSON kullaniliyor: {summary_path}")
        with open(summary_path) as f:
            summary = json.load(f)
        best_run = {'run_id': summary['best_run_id'],
                    'best_loss': summary['best_loss'],
                    'params': summary['best_params']}

    # =====================================================================
    # ASAMA 3b: Final GPU egitim (--skip-final degilse)
    # =====================================================================
    if args.skip_final:
        logger.info("\n[--skip-final] Final GPU egitim atlandi.")
        return 0

    logger.info("\n[4/4] FINAL EGITIM (GPU)")

    # Ablation flag'lerini dönüstür
    abl = args.ablation
    use_dwt = (abl != 'nodwt')
    use_bidirectional = (abl != 'nobidir')
    # waveletfixed env var ile parametre mapping etkilenecek; flag uretmiyoruz

    final_loss = run_final_training(algo, summary['best_params'],
                                      models_dir, figures_dir,
                                      use_dwt=use_dwt,
                                      use_bidirectional=use_bidirectional)

    # Sonucu summary'e ekle
    summary['final_training_loss'] = final_loss
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Final loss summary'e eklendi: {final_loss:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
