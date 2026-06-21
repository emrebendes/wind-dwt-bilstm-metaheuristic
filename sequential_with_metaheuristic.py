#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sequential_with_metaheuristic.py — Faz 3: Sequential vs Joint kiyasi.

Hakem 2.2 cevabi:
    "The motivation for adopting a joint optimization strategy over the commonly
    used sequential approach is reasonable. However, the need for this shift is
    primarily argued conceptually, and would benefit from stronger support."

Bilimsel cerceve:
    JOINT (Faz 1)  : 9 boyut birlikte optimize (wavelet, level, mode + BiLSTM 6D)
    SEQUENTIAL     : Asama 1 - Shannon entropy ile (wavelet, level, mode) sec
                     (dwt_param_selector.py, saniyeler)
                     Asama 2 - BiLSTM 6D ayni metasezgisel ile optimize edilir
                     (run_optimizer.py --fixed-wavelet X --fixed-level Y --fixed-mode Z)
                     Budget = joint ile esit (40 pop × 50 iter = 2000 eval)

Akis:
    1. Sequential Stage 1 — DWT parametrelerini sec (zaten bitti varsayilir):
       python dwt_param_selector.py
       -> dwt_selection_result.json (selected_wavelet, selected_level, selected_mode)

    2. Run dizinlerini + TRUBA submit scriptleri hazirla:
       python sequential_with_metaheuristic.py --prepare \\
              --algos ga,pso,fno --n-runs 30 \\
              --dwt-result dwt_selection_result.json

    3. TRUBA'da kostur (her algoritma icin tek script):
       bash submit_seq_ga.sh
       bash submit_seq_pso.sh
       bash submit_seq_fno.sh

    4. Durum kontrol:
       python sequential_with_metaheuristic.py --status --algos ga,pso,fno

    5. Analiz + joint kiyasi (tum kosular tamamlaninca):
       python sequential_with_metaheuristic.py --analyze \\
              --algos ga,pso,fno
"""

import os
import sys
import json
import time
import logging
import argparse
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [SEQ] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SEQ")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})


# =============================================================================
# 1. HAZIRLIK (--prepare)
# =============================================================================

def _load_phase1_seeds(n_runs: int) -> list:
    """Faz 1'in seed setini oku.

    Faz 1'de tum algoritmalar AYNI seed setini kullaniyor (gwo, ga, pso vb.
    hepsinde 670488, 288390, ... ayni). Bu yuzden herhangi bir algoritmadan
    okuyabiliriz. Sequential'in de AYNI seedleri kullanmasi paired Wilcoxon
    testini mumkun kilar (joint vs seq, ayni seed -> esleSmis fark).

    Returns:
        list[int]: run_001..run_{n_runs} sirasi ile seedler.
    """
    # Faz 1 dizini ara (gwo ilk tercih, sonra ga, sonra pso)
    candidates = ["gwo_runs", "ga_runs", "pso_runs", "fno_runs", "raindrop_runs",
                  "toa_runs", "ho_runs", "abc_runs"]
    for cand in candidates:
        cand_path = Path(cand)
        if cand_path.exists():
            seeds = []
            for r in range(1, n_runs + 1):
                cfg = cand_path / f"run_{r:03d}" / "config.json"
                if not cfg.exists():
                    return []  # Eksik run -> bos liste
                with open(cfg) as f:
                    seeds.append(json.load(f)["seed"])
            logger.info(f"Faz 1 seedleri {cand}'dan yuklendi (n={len(seeds)})")
            return seeds
    return []


def prepare_runs(algos: list, n_runs: int, dwt_result: dict,
                 pop_size: int, max_iter: int, workers: int) -> None:
    """Sequential run dizinleri + TRUBA submit scriptleri."""
    base = Path("sequential_runs")
    base.mkdir(exist_ok=True)
    logger.info(f"Hazirlik: algos={algos}, n_runs={n_runs}")
    logger.info(f"DWT sabit: wavelet={dwt_result['selected_wavelet']}, "
                f"level={dwt_result['selected_level']}, "
                f"mode={dwt_result['selected_mode']}")

    # Faz 1 seedlerini al -> paired Wilcoxon icin AYNI seed seti
    phase1_seeds = _load_phase1_seeds(n_runs)
    if not phase1_seeds:
        logger.error(
            f"Faz 1 seedleri yuklenemedi! {n_runs} run icin "
            f"<algo>_runs/run_XXX/config.json dosyalari eksik."
        )
        logger.error("Faz 1 dizinlerinin yerelde oldugundan emin ol.")
        return
    logger.info(f"Faz 1 seedleri: {phase1_seeds[:3]}...{phase1_seeds[-3:]} "
                f"(toplam {len(phase1_seeds)})")

    submit_scripts = []
    for algo in algos:
        algo_dir = base / f"{algo}_seq"
        algo_dir.mkdir(exist_ok=True)

        for run_id in range(1, n_runs + 1):
            run_dir = algo_dir / f"run_{run_id:03d}"
            run_dir.mkdir(exist_ok=True)
            seed = phase1_seeds[run_id - 1]  # Faz 1 ile AYNI seed
            config = {
                "algo": algo,
                "seed": seed,
                "pop_size": pop_size,
                "max_iter": max_iter,
                "workers": workers,
                "run_id": run_id,
                "fixed_wavelet": dwt_result['selected_wavelet'],
                "fixed_level":   dwt_result['selected_level'],
                "fixed_mode":    dwt_result['selected_mode'],
                "mode": "sequential",
                "dimension": 6,  # BiLSTM-only (look_back, hidden, layers, dropout, lr, batch)
            }
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        # TRUBA script
        script_path = Path(f"submit_seq_{algo}.sh")
        _write_truba_script(script_path, algo, n_runs)
        submit_scripts.append(script_path)
        logger.info(f"  {algo}: {n_runs} run dizini hazir -> {algo_dir}")

    logger.info(f"\nHazirlik tamamlandi. TRUBA'da kostur:")
    for sp in submit_scripts:
        logger.info(f"  bash {sp}")


def _write_truba_script(script_path: Path, algo: str, n_runs: int) -> None:
    script = f"""#!/bin/bash
#======================================================================
# Sequential TRUBA submitter: {algo}
# {n_runs} kosu paralel SLURM job olarak gonderilir.
#======================================================================

ALGO={algo}
N_RUNS={n_runs}

for i in $(seq 1 $N_RUNS); do
    RUN_ID=$(printf "%03d" $i)
    sbatch --job-name="seq_${{ALGO}}_${{RUN_ID}}" \\
           --output="sequential_runs/${{ALGO}}_seq/run_${{RUN_ID}}/slurm.out" \\
           --error="sequential_runs/${{ALGO}}_seq/run_${{RUN_ID}}/slurm.err" \\
           truba_sequential_one.sh $ALGO $i
    sleep 0.5
done

echo "Submitted $N_RUNS jobs for $ALGO (sequential)"
"""
    script_path.write_text(script)
    script_path.chmod(0o755)


# =============================================================================
# 2. TEK KOSU (--phase search; sbatch icinden cagrilir)
# =============================================================================

def run_single(algo: str, run_id: int) -> int:
    """Tek bir sequential HP search kosusunu calistir (TRUBA worker)."""
    run_dir = Path(f"sequential_runs/{algo}_seq/run_{run_id:03d}")
    config_path = run_dir / "config.json"
    if not config_path.exists():
        logger.error(f"Config yok: {config_path}")
        return 1
    with open(config_path) as f:
        config = json.load(f)

    db_path = run_dir / f"{algo}_running.db"
    logger.info(f"Tek kosu: {algo}/run_{run_id:03d}, seed={config['seed']}")
    logger.info(f"  Fixed DWT: wavelet={config['fixed_wavelet']}, "
                f"level={config['fixed_level']}, mode={config['fixed_mode']}")

    import subprocess
    cmd = [
        sys.executable, "run_optimizer.py",
        "--algo", algo,
        "--seed", str(config['seed']),
        "--db", str(db_path),
        "--pop-size", str(config['pop_size']),
        "--max-iter", str(config['max_iter']),
        "--workers", str(config.get("workers", 35)),
        "--fixed-wavelet", str(config['fixed_wavelet']),
        "--fixed-level",   str(config['fixed_level']),
        "--fixed-mode",    str(config['fixed_mode']),
    ]
    result = subprocess.run(cmd, check=False)
    return result.returncode


# =============================================================================
# 3. DURUM KONTROLU (--status)
# =============================================================================

def check_status(algos: list, n_runs: int) -> None:
    base = Path("sequential_runs")
    logger.info("Sequential durum:")
    for algo in algos:
        algo_dir = base / f"{algo}_seq"
        if not algo_dir.exists():
            logger.info(f"  {algo}: dizin yok")
            continue
        completed = 0
        running = 0
        for r in range(1, n_runs + 1):
            db = algo_dir / f"run_{r:03d}" / f"{algo}_running.db"
            if not db.exists():
                continue
            try:
                conn = sqlite3.connect(str(db))
                cur = conn.cursor()
                cur.execute("SELECT status FROM metadata")
                row = cur.fetchone()
                conn.close()
                if row and row[0] == "COMPLETED":
                    completed += 1
                else:
                    running += 1
            except Exception:
                pass
        logger.info(f"  {algo}: COMPLETED={completed}/{n_runs}, RUNNING={running}")


# =============================================================================
# 4b. FINAL TRAINING (--final; YEREL GPU)
# =============================================================================

def run_final_sequential(algo: str) -> float:
    """Sequential analizinden best HP alip TAM VERIYLE final training.

    Veri: 51144 saat (Y1..Y6 hepsi).
    Cihaz: GPU otomatik.
    Cikti: sequential_runs/{algo}_seq_analysis.json'a "final_training_loss" alani.
    """
    analysis_path = Path(f"sequential_runs/{algo}_seq_analysis.json")
    if not analysis_path.exists():
        logger.error(f"Once --analyze ile {analysis_path} olustur.")
        return float("inf")

    with open(analysis_path) as f:
        summary = json.load(f)

    best_params_str = summary["best_params"]
    # str-keys -> HP enum
    from utils import HP
    best_params = {}
    int_keys = (HP.DWT_LEVEL, HP.LOOK_BACK, HP.HIDDEN, HP.LAYERS, HP.BATCH)
    float_keys = (HP.DROPOUT, HP.LR)
    for hp in HP:
        if hp.value in best_params_str:
            v = best_params_str[hp.value]
            if hp in int_keys:
                v = int(v)
            elif hp in float_keys:
                v = float(v)
            best_params[hp] = v

    logger.info("=" * 60)
    logger.info(f"SEQUENTIAL FINAL TRAINING — {algo.upper()}")
    logger.info("=" * 60)
    logger.info(f"Best HP: {best_params_str}")
    logger.info(f"Veri: tam 51144 saat (Y1..Y6)")

    # Models/Figures redirect: algo-spesifik dizin
    import config as _cfg
    import objective as _obj_mod
    models_dir = Path(f"sequential_runs/{algo}_seq_final_models")
    figures_dir = Path(f"sequential_runs/{algo}_seq_final_figures")
    models_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)
    _cfg.MODELS_DIR_PATH = str(models_dir)
    _cfg.FIGURES_DIR_PATH = str(figures_dir)
    _obj_mod.MODELS_DIR_PATH = str(models_dir)
    _obj_mod.FIGURES_DIR_PATH = str(figures_dir)
    logger.info(f"Model dizini: {models_dir}")

    from objective import WindForecastObjective
    obj = WindForecastObjective(
        mode="global",
        is_final_training=True,
        show=10,
        use_gpu=None,        # otomatik GPU
        data_window=None,    # tam veri
    )
    final_loss = obj(best_params, log_prefix=f"[SEQ_FINAL_{algo.upper()}]")
    logger.info(f"Final loss: {final_loss:.6f}")

    summary["final_training_loss"] = float(final_loss)
    summary["final_training_at"] = datetime.now().isoformat()
    with open(analysis_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Guncellendi: {analysis_path}")
    return final_loss


# =============================================================================
# 4. ANALIZ (--analyze)
# =============================================================================

def analyze_sequential(algo: str) -> dict:
    """Bir algoritmanin sequential 30 kosusunu topla, best HP cikar."""
    algo_dir = Path(f"sequential_runs/{algo}_seq")
    if not algo_dir.exists():
        logger.error(f"Sequential dizin yok: {algo_dir}")
        return {}

    all_losses = []
    all_results = []
    run_dirs = sorted(algo_dir.glob("run_*"))

    for rd in run_dirs:
        db_path = rd / f"{algo}_running.db"
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("""
                SELECT loss, dwt_wavelet, dwt_level, dwt_mode, look_back,
                       hidden, layers, dropout, lr, batch
                FROM trial_history
                WHERE loss IS NOT NULL
                ORDER BY loss ASC LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                best_loss = float(row[0])
                params = {
                    "dwt_wavelet": row[1], "dwt_level": int(row[2]),
                    "dwt_mode": row[3], "look_back": int(row[4]),
                    "hidden": int(row[5]), "layers": int(row[6]),
                    "dropout": float(row[7]), "lr": float(row[8]),
                    "batch": int(row[9]),
                }
                all_losses.append(best_loss)
                all_results.append({
                    "run_id": int(rd.name.split("_")[1]),
                    "best_loss": best_loss,
                    "best_params": params,
                })
            conn.close()
        except Exception as e:
            logger.warning(f"DB okuma hatasi {db_path}: {e}")

    if not all_losses:
        return {}

    losses = np.array(all_losses)
    best_idx = int(np.argmin(losses))
    summary = {
        "algorithm": algo,
        "mode": "sequential",
        "n_runs_found": len(losses),
        "best_loss": all_results[best_idx]["best_loss"],
        "best_run_id": all_results[best_idx]["run_id"],
        "best_params": all_results[best_idx]["best_params"],
        "statistics": {
            "n_runs": len(losses),
            "min":    float(losses.min()),
            "max":    float(losses.max()),
            "mean":   float(losses.mean()),
            "median": float(np.median(losses)),
            "std":    float(losses.std()),
            "cv_percent": float(losses.std() / losses.mean() * 100),
        },
        "all_losses": losses.tolist(),
        "all_results": all_results,
        "created_at": datetime.now().isoformat(),
    }
    out_path = Path(f"sequential_runs/{algo}_seq_analysis.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def compare_joint_vs_sequential(algos: list, out_dir: Path) -> None:
    """Joint (Faz 1) ile Sequential (Faz 3) RMSE dagilimlarini kiyasla."""
    out_dir.mkdir(exist_ok=True)
    rows = []
    joint_data = {}
    seq_data = {}

    for algo in algos:
        # Joint sonucu Faz 1'den
        joint_path = Path(f"{algo}_final_results/analysis_summary.json")
        if joint_path.exists():
            with open(joint_path, "rb") as f:
                raw = f.read().replace(b"\x00", b"").strip()
            j = json.loads(raw)
            joint_data[algo] = j.get("all_losses", [])
            jbest = j.get("best_loss")
            jmean = j.get("statistics", {}).get("mean")
        else:
            jbest = jmean = None

        # Sequential sonucu
        seq_path = Path(f"sequential_runs/{algo}_seq_analysis.json")
        if seq_path.exists():
            with open(seq_path) as f:
                s = json.load(f)
            seq_data[algo] = s.get("all_losses", [])
            sbest = s.get("best_loss")
            smean = s.get("statistics", {}).get("mean")
        else:
            sbest = smean = None

        rows.append({
            "algo": algo,
            "joint_best": jbest,
            "joint_mean": jmean,
            "seq_best": sbest,
            "seq_mean": smean,
        })

    # Logla
    logger.info(f"\n=== JOINT vs SEQUENTIAL ===")
    logger.info(f"{'Algo':<10s} {'J_best':>10s} {'J_mean':>10s} {'S_best':>10s} {'S_mean':>10s}")
    for r in rows:
        logger.info(f"{r['algo']:<10s} "
                    f"{r['joint_best'] or 'NA':>10} "
                    f"{r['joint_mean'] or 'NA':>10} "
                    f"{r['seq_best'] or 'NA':>10} "
                    f"{r['seq_mean'] or 'NA':>10}")

    # Wilcoxon (algo basina, joint vs sequential 30 kosu)
    from scipy.stats import wilcoxon
    wilcoxon_results = {}
    for algo in algos:
        j_losses = joint_data.get(algo, [])
        s_losses = seq_data.get(algo, [])
        if len(j_losses) == 30 and len(s_losses) == 30:
            try:
                # PAIRED Wilcoxon signed-rank: ayni seed -> ayni run, eslestirilebilir.
                # H0: medyan fark = 0 (joint = sequential)
                # H1: joint < seq (joint daha iyi) veya tersi
                stat, p = wilcoxon(j_losses, s_losses, alternative="two-sided",
                                   zero_method="wilcox")
                wilcoxon_results[algo] = {
                    "test": "wilcoxon_signed_rank_paired",
                    "stat": float(stat),
                    "p_value": float(p),
                    "joint_median": float(np.median(j_losses)),
                    "seq_median": float(np.median(s_losses)),
                    "joint_better": float(np.median(j_losses)) < float(np.median(s_losses)),
                    "significant_at_005": bool(p < 0.05),
                }
                logger.info(f"\n{algo}: Paired Wilcoxon p={p:.4f}, "
                            f"joint median={np.median(j_losses):.5f}, "
                            f"seq median={np.median(s_losses):.5f}")
            except Exception as e:
                wilcoxon_results[algo] = {"error": str(e)}

    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data, box_labels = [], []
    for algo in algos:
        if joint_data.get(algo):
            box_data.append(joint_data[algo])
            box_labels.append(f"{algo.upper()}\nJoint")
        if seq_data.get(algo):
            box_data.append(seq_data[algo])
            box_labels.append(f"{algo.upper()}\nSeq")
    if box_data:
        ax.boxplot(box_data, labels=box_labels)
        ax.set_ylabel("HP search RMSE")
        ax.set_title("Joint (Faz 1) vs Sequential (Faz 3) — 30 koSu dagilimi")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "joint_vs_sequential_boxplot.png")
        plt.close(fig)
        logger.info(f"\nBoxplot: {out_dir / 'joint_vs_sequential_boxplot.png'}")

    # LaTeX tablo
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Joint vs Sequential optimization comparison (Faz 1 vs Faz 3)}",
        r"\label{tab:joint_vs_seq}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Algorithm & Joint min & Joint mean & Seq min & Seq mean & MWU p-val & Better \\",
        r"\hline",
    ]
    for r in rows:
        algo = r['algo']
        wr = wilcoxon_results.get(algo, {})
        p = wr.get("p_value", float('nan'))
        better = "Joint" if wr.get("joint_better") else ("Seq" if "joint_better" in wr else "-")
        sig = "*" if wr.get("significant_at_005") else ""
        tex_lines.append(
            f"{algo.upper():<8s} & "
            f"{r['joint_best']:.5f} & {r['joint_mean']:.5f} & "
            f"{r['seq_best']:.5f} & {r['seq_mean']:.5f} & "
            f"{p:.3f}{sig} & {better} \\\\"
        )
    tex_lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    tex_path = out_dir / "joint_vs_sequential.tex"
    tex_path.write_text("\n".join(tex_lines))
    logger.info(f"LaTeX: {tex_path}")

    # Toplu JSON
    summary = {
        "algos": algos,
        "rows": rows,
        "wilcoxon": wilcoxon_results,
        "created_at": datetime.now().isoformat(),
    }
    out_json = out_dir / "joint_vs_sequential_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Ozet: {out_json}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--search", action="store_true",
                        help="Tek kosu (sbatch icinden)")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--final", action="store_true",
                        help="Best HP ile tam veriyle YEREL GPU final eGitim "
                             "(tek algo, --algo ile)")
    parser.add_argument("--analyze", action="store_true")

    parser.add_argument("--algos", type=str, default="ga,pso,fno",
                        help="Algoritma listesi virgulle, default: ga,pso,fno")
    parser.add_argument("--algo", type=str, default=None,
                        help="Tek algo (--search icin)")
    parser.add_argument("--run-id", type=int, default=None)

    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=40)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--workers", type=int, default=40)

    parser.add_argument("--dwt-result", type=str,
                        default="dwt_selection_result.json",
                        help="dwt_param_selector.py ciktisi")
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    if args.prepare:
        dwt_path = Path(args.dwt_result)
        if not dwt_path.exists():
            logger.error(f"Once dwt_param_selector.py kostur: {dwt_path} yok")
            return 1
        with open(dwt_path) as f:
            dwt_result = json.load(f)
        prepare_runs(algos, args.n_runs, dwt_result,
                     args.pop_size, args.max_iter, args.workers)
        return 0

    if args.search:
        if not args.algo or args.run_id is None:
            parser.error("--search: --algo ve --run-id zorunlu")
        return run_single(args.algo, args.run_id)

    if args.status:
        check_status(algos, args.n_runs)
        return 0

    if args.final:
        if args.algo is None:
            parser.error("--final: --algo zorunlu (tek algoritma)")
        run_final_sequential(args.algo)
        return 0

    if args.analyze:
        for algo in algos:
            s = analyze_sequential(algo)
            if s:
                logger.info(f"\n[{algo}] best={s['best_loss']:.6f}, "
                            f"mean={s['statistics']['mean']:.6f}, "
                            f"n={s['n_runs_found']}")
        compare_joint_vs_sequential(algos, Path("sequential_compare"))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
