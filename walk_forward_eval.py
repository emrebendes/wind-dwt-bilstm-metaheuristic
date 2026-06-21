#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
walk_forward_eval.py — Faz 2: Walk-forward CV (algoritma-agnostik).

Hakem 2.6 ve Hakem 3.Q4 cevabi:
    Y6'da (son 1 yil) HP search yapildi (Faz 1) -> robust mu?
    Y2..Y5'te de ayni protokol HP search + tam veri final training
    -> sonuclar tutarli mi?

Pencere semasi (8760 saat = 1 yil):
    Y6: hours [42384:51144]  (Faz 1, halihazirda yapildi)
    Y5: hours [33624:42384]
    Y4: hours [24864:33624]
    Y3: hours [16104:24864]
    Y2: hours [ 7344:16104]
    Y1: hours [    0: 7344]  (10 ay, HP search DISI; final training'de var)

Pipeline:
    1) --phase prepare  : run dizinleri + SLURM scriptleri uretir (her pencere icin)
    2) --phase search   : tek bir kosu (TRUBA worker) — run_optimizer.py wrapper
    3) --phase analyze-window : bir pencerenin 30 kosusunu topla, best HP cikar
    4) --phase final    : pencerenin best HP'siyle tam veri final training (yerel GPU)
    5) --phase compare  : tum pencereler arasi karsilastirma (analyze_walk_forward.py'a delege)

Kullanim:
    # 1) Hazirlik (run dizinleri + TRUBA scriptleri)
    python walk_forward_eval.py --phase prepare --algo ga --windows Y2,Y3,Y4,Y5 \\
        --n-runs 30 --pop-size 40 --max-iter 50

    # 2) TRUBA'da kostur (uretilen scriptlerden biri)
    bash submit_wf_ga_Y2.sh    # icinde sbatch komutlari

    # 3) Bir pencerenin tum kosulari bitince
    python walk_forward_eval.py --phase analyze-window --algo ga --window Y2

    # 4) Best HP ile tam veri final training (yerel GPU)
    python walk_forward_eval.py --phase final --algo ga --window Y2

    # 5) Tum pencereler bitince karsilastirma
    python walk_forward_eval.py --phase compare --algo ga --windows Y2,Y3,Y4,Y5
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

# Thread limit (TRUBA uyumlulugu)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [WF] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("WF")


# =============================================================================
# PENCERE TANIMI
# =============================================================================

# 8760 saat = 1 yil. Veri: 51144 saat (5.84 yil).
# Y6 = Faz 1 (son 1 yil, halihazirda yapildi). Y1 = ilk 7344 saat (10 ay, atilir).
WINDOWS = {
    "Y1": (0,      7344),    # 10 ay - HP search disi (kisa, ardilik bias)
    "Y2": (7344,  16104),    # 8760 saat
    "Y3": (16104, 24864),    # 8760 saat
    "Y4": (24864, 33624),    # 8760 saat
    "Y5": (33624, 42384),    # 8760 saat
    "Y6": (42384, 51144),    # 8760 saat - FAZ 1 SONUCLARI
}

# Faz 1 sonuclarinin nerede oldugu (algoritma -> klasor)
PHASE1_RESULTS_DIR = "{algo}_final_results"  # ornek: "ga_final_results"


# =============================================================================
# HAZIRLIK (--phase prepare)
# =============================================================================

def prepare_runs(algo: str, windows: list, n_runs: int,
                 pop_size: int, max_iter: int, workers: int) -> None:
    """Walk-forward run dizinlerini olustur ve TRUBA submit scriptleri uret.

    Cikti: walk_forward_runs/{algo}_{window}/run_{001..030}/config.json
           submit_wf_{algo}_{window}.sh  (her pencere icin tek dosya)
    """
    base = Path("walk_forward_runs")
    base.mkdir(exist_ok=True)
    logger.info(f"Hazirlik baslatildi: algo={algo}, windows={windows}, n_runs={n_runs}")

    submit_scripts = []

    for win_name in windows:
        if win_name not in WINDOWS:
            logger.error(f"Bilinmeyen pencere: {win_name}. {list(WINDOWS.keys())}")
            continue

        w_start, w_end = WINDOWS[win_name]
        win_dir = base / f"{algo}_{win_name}"
        win_dir.mkdir(exist_ok=True)

        # Her run icin alt dizin + config.json (seed her run'da farkli)
        for run_id in range(1, n_runs + 1):
            run_dir = win_dir / f"run_{run_id:03d}"
            run_dir.mkdir(exist_ok=True)
            # Seed: pencere * 1000 + run -> her pencerede ayni seed seti
            seed = 42000 + run_id  # Faz 1 ile ayni mantik
            config = {
                "algo": algo,
                "window": win_name,
                "window_start": w_start,
                "window_end": w_end,
                "seed": seed,
                "pop_size": pop_size,
                "max_iter": max_iter,
                "workers": workers,
                "run_id": run_id,
            }
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        # TRUBA submit scripti
        script_path = Path(f"submit_wf_{algo}_{win_name}.sh")
        _write_truba_script(script_path, algo, win_name, n_runs)
        submit_scripts.append(script_path)

        logger.info(f"  {win_name}: {n_runs} run dizini hazir -> {win_dir}")

    logger.info(f"Hazirlik tamamlandi. TRUBA'da bunlari kostur:")
    for sp in submit_scripts:
        logger.info(f"  bash {sp}")


def _write_truba_script(script_path: Path, algo: str, win_name: str,
                        n_runs: int) -> None:
    """Tek pencere icin tum koSulari TRUBA'ya gonderecek script."""
    script = f"""#!/bin/bash
#======================================================================
# Walk-forward TRUBA submitter: {algo} / {win_name}
# {n_runs} kosu paralel SLURM job olarak gonderilir.
# Her job: 36-core CPU node, 1 hp search kosusu.
#======================================================================

ALGO={algo}
WINDOW={win_name}
N_RUNS={n_runs}

for i in $(seq 1 $N_RUNS); do
    RUN_ID=$(printf "%03d" $i)
    sbatch --job-name="wf_${{ALGO}}_${{WINDOW}}_${{RUN_ID}}" \\
           --output="walk_forward_runs/${{ALGO}}_${{WINDOW}}/run_${{RUN_ID}}/slurm.out" \\
           --error="walk_forward_runs/${{ALGO}}_${{WINDOW}}/run_${{RUN_ID}}/slurm.err" \\
           truba_walkforward_one.sh $ALGO $WINDOW $i
    sleep 0.5
done

echo "Submitted $N_RUNS jobs for $ALGO / $WINDOW"
"""
    script_path.write_text(script)
    script_path.chmod(0o755)


# =============================================================================
# TEK KOSU (--phase search; sbatch icinden cagrilir)
# =============================================================================

def run_single(algo: str, window: str, run_id: int) -> int:
    """Tek bir HP search kosusunu calistir (TRUBA worker icinden cagrilir)."""
    run_dir = Path(f"walk_forward_runs/{algo}_{window}/run_{run_id:03d}")
    config_path = run_dir / "config.json"
    if not config_path.exists():
        logger.error(f"Config yok: {config_path}")
        return 1
    with open(config_path) as f:
        config = json.load(f)

    db_path = run_dir / f"{algo}_running.db"
    logger.info(f"Tek kosu: {algo}/{window}/run_{run_id:03d}, seed={config['seed']}")

    # run_optimizer.py'i alt-process olarak cagiriyoruz (subprocess yerine
    # direkt main() cagrisini tercih edebiliriz; subprocess gibi sleep yok)
    import subprocess
    cmd = [
        sys.executable, "run_optimizer.py",
        "--algo", algo,
        "--seed", str(config['seed']),
        "--db", str(db_path),
        "--pop-size", str(config['pop_size']),
        "--max-iter", str(config['max_iter']),
        "--workers", str(config.get("workers", 35)),
        "--data-window-start", str(config['window_start']),
        "--data-window-end", str(config['window_end']),
    ]
    result = subprocess.run(cmd, check=False)
    return result.returncode


# =============================================================================
# PENCERE ANALIZ (--phase analyze-window)
# =============================================================================

def analyze_window(algo: str, window: str) -> dict:
    """Bir pencerenin 30 kosusunu tarayip best HP'yi cikar.

    Cikti: walk_forward_runs/{algo}_{window}_analysis.json
    """
    if window not in WINDOWS:
        logger.error(f"Bilinmeyen pencere: {window}")
        return {}

    win_dir = Path(f"walk_forward_runs/{algo}_{window}")
    if not win_dir.exists():
        logger.error(f"Pencere dizini yok: {win_dir}")
        return {}

    # Tum run DB'lerini tara
    import sqlite3
    all_losses = []
    all_results = []
    run_dirs = sorted(win_dir.glob("run_*"))

    for rd in run_dirs:
        db_path = rd / f"{algo}_running.db"
        if not db_path.exists():
            logger.warning(f"DB yok: {db_path}")
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            # Schema: trial_history tablo, HP'ler ayri kolonlar
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
                    "dwt_wavelet": row[1],
                    "dwt_level":   int(row[2]),
                    "dwt_mode":    row[3],
                    "look_back":   int(row[4]),
                    "hidden":      int(row[5]),
                    "layers":      int(row[6]),
                    "dropout":     float(row[7]),
                    "lr":          float(row[8]),
                    "batch":       int(row[9]),
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
        logger.error(f"Hicbir gecerli sonuc bulunamadi: {algo}/{window}")
        return {}

    losses = np.array(all_losses)
    best_idx = int(np.argmin(losses))
    summary = {
        "algorithm": algo,
        "window": window,
        "window_hours": WINDOWS[window],
        "n_runs_found": len(all_losses),
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
        },
        "all_losses": losses.tolist(),
        "all_results": all_results,
        "created_at": datetime.now().isoformat(),
    }

    out_path = win_dir.parent / f"{algo}_{window}_analysis.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"[{window}] {algo.upper()}: best={summary['best_loss']:.6f} "
        f"(run {summary['best_run_id']}), mean={losses.mean():.6f}, "
        f"std={losses.std():.6f}, n={len(losses)}"
    )
    logger.info(f"Best HP: {summary['best_params']}")
    logger.info(f"Kaydedildi: {out_path}")
    return summary


# =============================================================================
# FINAL TRAINING (--phase final; YEREL GPU)
# =============================================================================

def run_final(algo: str, window: str) -> float:
    """Bir pencerenin best HP'siyle TAM VERI uzerinde final training.

    Veri: tum 51144 saat (Y1+Y2+...+Y6 hepsi, Y1 dahil).
    Cihaz: GPU (yerel 6 GiB).
    """
    analysis_path = Path(f"walk_forward_runs/{algo}_{window}_analysis.json")
    if not analysis_path.exists():
        logger.error(f"Once --phase analyze-window calistir: {analysis_path}")
        return float("inf")

    with open(analysis_path) as f:
        summary = json.load(f)

    best_params_str = summary["best_params"]
    # str -> HP enum dict
    from utils import HP
    best_params = {}
    for hp in HP:
        key = hp.value
        if key in best_params_str:
            val = best_params_str[key]
            if hp in (HP.DWT_LEVEL, HP.LOOK_BACK, HP.HIDDEN, HP.LAYERS, HP.BATCH):
                val = int(val)
            elif hp in (HP.DROPOUT, HP.LR):
                val = float(val)
            best_params[hp] = val

    logger.info(f"=" * 60)
    logger.info(f"FINAL TRAINING — {algo.upper()} / {window}")
    logger.info(f"=" * 60)
    logger.info(f"HP: {best_params_str}")
    logger.info(f"Veri: tam 51144 saat (Y1+...+Y6)")
    logger.info(f"Cihaz: GPU (otomatik)")

    # Models/Figures redirect: KAYBOLMASIN diye pencere-spesifik dizinler
    import config as _cfg
    import objective as _obj_mod
    models_dir = Path(f"walk_forward_runs/{algo}_{window}_final_models")
    figures_dir = Path(f"walk_forward_runs/{algo}_{window}_final_figures")
    models_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)
    _cfg.MODELS_DIR_PATH = str(models_dir)
    _cfg.FIGURES_DIR_PATH = str(figures_dir)
    _obj_mod.MODELS_DIR_PATH = str(models_dir)
    _obj_mod.FIGURES_DIR_PATH = str(figures_dir)
    logger.info(f"Model dizini: {models_dir}")
    logger.info(f"Figur dizini: {figures_dir}")

    # WindForecastObjective is_final=True ile cagir (otomatik GPU + tam veri)
    from objective import WindForecastObjective
    obj = WindForecastObjective(
        mode="global",
        is_final_training=True,
        show=10,
        use_gpu=None,         # otomatik: cuda varsa GPU
        data_window=None,     # final her zaman tam veri
    )
    final_loss = obj(best_params, log_prefix=f"[WF_FINAL_{algo.upper()}_{window}]")
    logger.info(f"Final loss: {final_loss:.6f}")

    # Sonuca kaydet
    summary["final_training_loss"] = float(final_loss)
    summary["final_training_at"] = datetime.now().isoformat()
    with open(analysis_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Guncellendi: {analysis_path}")
    return final_loss


# =============================================================================
# KARSILASTIRMA (--phase compare)
# =============================================================================

def compare_windows(algo: str, windows: list) -> None:
    """Tum pencereleri (Y2..Y5 + Y6) karsilastir.

    analyze_walk_forward.py'a delege eder.
    """
    import subprocess
    cmd = [sys.executable, "analyze_walk_forward.py",
           "--algo", algo,
           "--windows", ",".join(windows + ["Y6"])]
    subprocess.run(cmd, check=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward CV — algoritma-agnostik",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--phase", required=True,
                        choices=["prepare", "search", "analyze-window",
                                 "final", "compare"],
                        help="Pipeline asamasi")
    parser.add_argument("--algo", required=True,
                        help="Algoritma adi (ga, fno, raindrop, toa, ...)")
    parser.add_argument("--windows", default="Y2,Y3,Y4,Y5",
                        help="Pencere listesi virgulle, ornek: Y2,Y3,Y4,Y5")
    parser.add_argument("--window", default=None,
                        help="Tek pencere (analyze-window/final/search icin)")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Run ID (search asamasi icin)")
    parser.add_argument("--n-runs", type=int, default=30,
                        help="Pencere basi kosu sayisi (default: 30)")
    parser.add_argument("--pop-size", type=int, default=40,
                        help="Pop size (default: 40, Faz 1 ile ayni)")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Max iter (default: 50, Faz 1 ile ayni)")
    parser.add_argument("--workers", type=int, default=40,
                        help="HP search icin CPU worker sayisi "
                             "(default: 35; TRUBA 36-core node icin. "
                             "Yerel PC'de daha az olabilir)")
    args = parser.parse_args()

    windows = [w.strip() for w in args.windows.split(",") if w.strip()]

    if args.phase == "prepare":
        prepare_runs(args.algo, windows, args.n_runs,
                     args.pop_size, args.max_iter, args.workers)
        return 0

    if args.phase == "search":
        if args.window is None or args.run_id is None:
            parser.error("--phase search: --window ve --run-id zorunlu")
        return run_single(args.algo, args.window, args.run_id)

    if args.phase == "analyze-window":
        if args.window is None:
            parser.error("--phase analyze-window: --window zorunlu")
        analyze_window(args.algo, args.window)
        return 0

    if args.phase == "final":
        if args.window is None:
            parser.error("--phase final: --window zorunlu")
        run_final(args.algo, args.window)
        return 0

    if args.phase == "compare":
        compare_windows(args.algo, windows)
        return 0


if __name__ == "__main__":
    sys.exit(main())
