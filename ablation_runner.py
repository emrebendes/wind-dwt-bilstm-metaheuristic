#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ablation_runner.py — Faz 4 Ablation orkestratoru.

Hakem 1.13 cevabi:
    "The results section should include statistical significance tests and,
    if possible, ablation studies to validate the contribution of each component."

Bilimsel cerceve (Joint Optimization mimari bilesenlerinin katki testi):

    BASELINE (Faz 1)        : Full pipeline (DWT + BiLSTM, joint 9D)
                              -> {algo}_runs/  (ZATEN VAR)

    ABLATION 1 (nodwt)      : DWT KAPALI (ham sinyal direkt BiLSTM)
                              dwt_wavelet/level/mode optimize edilmez
                              -> ablation_runs/{algo}_nodwt/

    ABLATION 2 (nobidir)    : BiLSTM yerine duz LSTM (forward only)
                              -> ablation_runs/{algo}_nobidir/

    (Opsiyonel ABLATION 3: waveletfixed — wavelet sym5 sabit, sadece level+mode arama)

Her senaryo icin 30 koSu Faz 1 SEEDLERI ile (paired Wilcoxon yapilabilsin).
Compare_ablations.py'da Mann-Whitney U / Friedman testi ile farklar yorumlanir.

Akis:
    1. Run dizinleri + TRUBA submit scriptleri:
       python ablation_runner.py --prepare --algo gwo \\
              --ablations nodwt,nobidir --n-runs 30

    2. TRUBA'da kostur (her ablation icin 1 script):
       bash submit_abl_gwo_nodwt.sh
       bash submit_abl_gwo_nobidir.sh

    3. Durum:
       python ablation_runner.py --status --algo gwo --ablations nodwt,nobidir

    4. Analiz:
       python compare_ablations.py --algo gwo --ablations nodwt,nobidir
"""

import os
import sys
import json
import logging
import argparse
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ABL] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ABL")


# Desteklenen ablation senaryolari
ABLATIONS = {
    "nodwt":        "DWT KAPALI — ham sinyal direkt BiLSTM (level=1)",
    "nobidir":      "BiLSTM yerine duz LSTM (forward only)",
    "waveletfixed": "Wavelet sym5 sabit (level/mode optimize)",
}


# =============================================================================
# 1. HAZIRLIK (--prepare)
# =============================================================================

def _load_phase1_seeds(n_runs: int) -> list:
    """Faz 1'in seed setini oku (paired Wilcoxon icin AYNI seedler)."""
    candidates = ["gwo_runs", "ga_runs", "pso_runs", "fno_runs",
                  "raindrop_runs", "toa_runs", "ho_runs", "abc_runs"]
    for cand in candidates:
        cand_path = Path(cand)
        if cand_path.exists():
            seeds = []
            for r in range(1, n_runs + 1):
                cfg = cand_path / f"run_{r:03d}" / "config.json"
                if not cfg.exists():
                    return []
                with open(cfg) as f:
                    seeds.append(json.load(f)["seed"])
            logger.info(f"Faz 1 seedleri {cand}'dan yuklendi (n={len(seeds)})")
            return seeds
    return []


def prepare_runs(algo: str, ablations: list, n_runs: int,
                 pop_size: int, max_iter: int, workers: int) -> None:
    """Ablation run dizinleri + TRUBA submit scriptleri."""
    base = Path("ablation_runs")
    base.mkdir(exist_ok=True)
    logger.info(f"Hazirlik: algo={algo}, ablations={ablations}, n_runs={n_runs}")

    # Faz 1 seedleri
    phase1_seeds = _load_phase1_seeds(n_runs)
    if not phase1_seeds:
        logger.error(f"Faz 1 seedleri yuklenemedi (gwo_runs vb. yok mu?)")
        return
    logger.info(f"Faz 1 seedleri: {phase1_seeds[:3]}...{phase1_seeds[-3:]} "
                f"(n={len(phase1_seeds)})")

    submit_scripts = []
    for abl in ablations:
        if abl not in ABLATIONS:
            logger.error(f"Bilinmeyen ablation: {abl}. Desteklenen: {list(ABLATIONS.keys())}")
            continue

        abl_dir = base / f"{algo}_{abl}"
        abl_dir.mkdir(exist_ok=True)
        logger.info(f"  {abl}: {ABLATIONS[abl]}")

        for run_id in range(1, n_runs + 1):
            run_dir = abl_dir / f"run_{run_id:03d}"
            run_dir.mkdir(exist_ok=True)
            seed = phase1_seeds[run_id - 1]   # Faz 1 ile AYNI seed
            config = {
                "algo": algo,
                "ablation": abl,
                "ablation_desc": ABLATIONS[abl],
                "seed": seed,
                "pop_size": pop_size,
                "max_iter": max_iter,
                "workers": workers,
                "run_id": run_id,
            }
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        # TRUBA submit
        script_path = Path(f"submit_abl_{algo}_{abl}.sh")
        _write_truba_script(script_path, algo, abl, n_runs)
        submit_scripts.append(script_path)
        logger.info(f"    {n_runs} run dizini hazir -> {abl_dir}")

    logger.info(f"\nHazirlik tamamlandi. TRUBA'da kostur:")
    for sp in submit_scripts:
        logger.info(f"  bash {sp}")


def _write_truba_script(script_path: Path, algo: str, abl: str, n_runs: int) -> None:
    script = f"""#!/bin/bash
#======================================================================
# Ablation TRUBA submitter: {algo} / {abl}
# {n_runs} kosu paralel SLURM job olarak gonderilir.
#======================================================================

ALGO={algo}
ABLATION={abl}
N_RUNS={n_runs}

for i in $(seq 1 $N_RUNS); do
    RUN_ID=$(printf "%03d" $i)
    sbatch --job-name="abl_${{ALGO}}_${{ABLATION}}_${{RUN_ID}}" \\
           --output="ablation_runs/${{ALGO}}_${{ABLATION}}/run_${{RUN_ID}}/slurm.out" \\
           --error="ablation_runs/${{ALGO}}_${{ABLATION}}/run_${{RUN_ID}}/slurm.err" \\
           truba_ablation_one.sh $ALGO $ABLATION $i
    sleep 0.5
done

echo "Submitted $N_RUNS jobs for $ALGO / $ABLATION"
"""
    script_path.write_text(script)
    script_path.chmod(0o755)


# =============================================================================
# 2. TEK KOSU (--search; sbatch icinden cagrilir)
# =============================================================================

def run_single(algo: str, ablation: str, run_id: int) -> int:
    """Tek ablation kosusu (TRUBA worker'dan cagrilir)."""
    run_dir = Path(f"ablation_runs/{algo}_{ablation}/run_{run_id:03d}")
    config_path = run_dir / "config.json"
    if not config_path.exists():
        logger.error(f"Config yok: {config_path}")
        return 1
    with open(config_path) as f:
        config = json.load(f)

    db_path = run_dir / f"{algo}_running.db"
    logger.info(f"Tek kosu: {algo}/{ablation}/run_{run_id:03d}, seed={config['seed']}")
    logger.info(f"  Ablation: {ABLATIONS.get(ablation, ablation)}")

    import subprocess
    cmd = [
        sys.executable, "run_optimizer.py",
        "--algo", algo,
        "--seed", str(config['seed']),
        "--db", str(db_path),
        "--pop-size", str(config['pop_size']),
        "--max-iter", str(config['max_iter']),
        "--workers", str(config.get("workers", 35)),
        "--ablation", ablation,
    ]
    result = subprocess.run(cmd, check=False)
    return result.returncode


# =============================================================================
# 3. DURUM (--status)
# =============================================================================

def check_status(algo: str, ablations: list, n_runs: int) -> None:
    base = Path("ablation_runs")
    logger.info("Ablation durum:")
    for abl in ablations:
        abl_dir = base / f"{algo}_{abl}"
        if not abl_dir.exists():
            logger.info(f"  {algo}/{abl}: dizin yok")
            continue
        completed = 0
        running = 0
        for r in range(1, n_runs + 1):
            db = abl_dir / f"run_{r:03d}" / f"{algo}_running.db"
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
        logger.info(f"  {algo}/{abl}: COMPLETED={completed}/{n_runs}, RUNNING={running}")


# =============================================================================
# 3b. PENCERE ANALIZI (--analyze-window)
# =============================================================================

def analyze_ablation(algo: str, ablation: str) -> dict:
    """Bir ablation'in 30 koSusunu topla, best HP cikar.

    Cikti: ablation_runs/{algo}_{ablation}_analysis.json
    """
    abl_dir = Path(f"ablation_runs/{algo}_{ablation}")
    if not abl_dir.exists():
        logger.error(f"Ablation dizin yok: {abl_dir}")
        return {}

    import numpy as np
    all_losses = []
    all_results = []
    run_dirs = sorted(abl_dir.glob("run_*"))
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
                FROM trial_history WHERE loss IS NOT NULL
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
            logger.warning(f"DB hatasi {db_path}: {e}")

    if not all_losses:
        return {}

    losses = np.array(all_losses)
    best_idx = int(np.argmin(losses))
    summary = {
        "algorithm": algo,
        "ablation": ablation,
        "n_runs_found": len(losses),
        "best_loss": all_results[best_idx]["best_loss"],
        "best_run_id": all_results[best_idx]["run_id"],
        "best_params": all_results[best_idx]["best_params"],
        "statistics": {
            "min":    float(losses.min()),
            "max":    float(losses.max()),
            "mean":   float(losses.mean()),
            "median": float(np.median(losses)),
            "std":    float(losses.std()),
        },
        "all_losses": losses.tolist(),
        "created_at": datetime.now().isoformat(),
    }
    out_path = abl_dir.parent / f"{algo}_{ablation}_analysis.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[{ablation}] {algo.upper()}: best={summary['best_loss']:.6f}, "
                f"mean={losses.mean():.6f}, std={losses.std():.6f}, n={len(losses)}")
    logger.info(f"  Best HP: {summary['best_params']}")
    logger.info(f"Kaydedildi: {out_path}")
    return summary


# =============================================================================
# 3c. FINAL TRAINING (--final; YEREL GPU)
# =============================================================================

def run_final_ablation(algo: str, ablation: str) -> float:
    """Bir ablation'in best HP'siyle TAM VERIYLE final training (ablation flag aktif).

    Veri: 51144 saat. Cihaz: GPU.
    use_dwt / use_bidirectional bayraklari ablation'a gore set edilir.
    """
    analysis_path = Path(f"ablation_runs/{algo}_{ablation}_analysis.json")
    if not analysis_path.exists():
        logger.error(f"Once --analyze-window calistir: {analysis_path} yok")
        return float("inf")

    with open(analysis_path) as f:
        summary = json.load(f)

    best_params_str = summary["best_params"]
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

    # Ablation flagleri
    use_dwt = (ablation != "nodwt")
    use_bidir = (ablation != "nobidir")

    logger.info("=" * 60)
    logger.info(f"ABLATION FINAL TRAINING — {algo.upper()} / {ablation}")
    logger.info("=" * 60)
    logger.info(f"use_dwt={use_dwt}, use_bidirectional={use_bidir}")
    logger.info(f"Best HP: {best_params_str}")

    # Models/Figures redirect: ablation-spesifik dizin
    import config as _cfg
    import objective as _obj_mod
    models_dir = Path(f"ablation_runs/{algo}_{ablation}_final_models")
    figures_dir = Path(f"ablation_runs/{algo}_{ablation}_final_figures")
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
        use_gpu=None,
        use_dwt=use_dwt,
        use_bidirectional=use_bidir,
        data_window=None,
    )
    final_loss = obj(best_params, log_prefix=f"[ABL_FINAL_{algo.upper()}_{ablation}]")
    logger.info(f"Final loss: {final_loss:.6f}")

    summary["final_training_loss"] = float(final_loss)
    summary["final_training_at"] = datetime.now().isoformat()
    with open(analysis_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Guncellendi: {analysis_path}")
    return final_loss


# =============================================================================
# 4. ANALIZ (--analyze)  — compare_ablations.py'a delege
# =============================================================================

def run_analyze(algo: str, ablations: list) -> None:
    import subprocess
    cmd = [sys.executable, "compare_ablations.py",
           "--algo", algo,
           "--ablations", ",".join(ablations)]
    logger.info(f"Cagriliyor: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


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
                        help="Tek koSu (sbatch icinden)")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--analyze-window", action="store_true",
                        help="Tek ablation'in 30 koSusunu topla, best HP cikar "
                             "(--ablation gerek)")
    parser.add_argument("--final", action="store_true",
                        help="Best HP ile tam veriyle YEREL GPU final eGitim "
                             "(--ablation gerek)")
    parser.add_argument("--analyze", action="store_true",
                        help="compare_ablations.py'a delege (toplu kiyas)")

    parser.add_argument("--algo", required=True,
                        help="Algoritma adi (gwo, ga, pso, ...)")
    parser.add_argument("--ablations", type=str, default="nodwt,nobidir",
                        help="Ablation listesi virgulle. Desteklenen: "
                             f"{','.join(ABLATIONS.keys())}")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Tek ablation (--search icin)")
    parser.add_argument("--run-id", type=int, default=None)

    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=40)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--workers", type=int, default=35)
    args = parser.parse_args()

    abls = [a.strip() for a in args.ablations.split(",") if a.strip()]

    if args.prepare:
        prepare_runs(args.algo, abls, args.n_runs,
                     args.pop_size, args.max_iter, args.workers)
        return 0
    if args.search:
        if args.ablation is None or args.run_id is None:
            parser.error("--search: --ablation ve --run-id zorunlu")
        return run_single(args.algo, args.ablation, args.run_id)
    if args.status:
        check_status(args.algo, abls, args.n_runs)
        return 0
    # argparse'da --analyze-window -> analyze_window oluyor
    if getattr(args, "analyze_window", False):
        if args.ablation is None:
            parser.error("--analyze-window: --ablation zorunlu (tek ablation)")
        analyze_ablation(args.algo, args.ablation)
        return 0
    if args.final:
        if args.ablation is None:
            parser.error("--final: --ablation zorunlu (tek ablation)")
        run_final_ablation(args.algo, args.ablation)
        return 0
    if args.analyze:
        run_analyze(args.algo, abls)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
