#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_optimizer.py — Birleşik metasezgisel optimizer CLI.

Tüm algoritmaları tek script üzerinden çalıştırır. Eski
run_abc_optimization_v2.py ve run_paralel_ga.py dosyalarının yerini alır
ama onlar dokunulmadan kalır (geriye dönük uyumluluk).

Kullanım:
    # ABC çalıştır
    python run_optimizer.py --algo abc --seed 42 --db abc_runs/run_001/abc_running.db

    # GA çalıştır
    python run_optimizer.py --algo ga --seed 42 --db ga_runs/run_001/ga_running.db

    # Mevcut algoritmaları listele
    python run_optimizer.py --list-algos

    # Sequential mode (deneysel)
    python run_optimizer.py --algo abc --mode sequential --seed 42 --db ...

    # Algoritma-spesifik parametreleri override et (JSON)
    python run_optimizer.py --algo abc --algo-params '{"limit": 7}' --seed 42 --db ...

TRUBA SLURM kullanımı:
    Eski SLURM scriptleri içinde
        python run_abc_optimization_v2.py --seed $SEED --db $DB
    yerine
        python run_optimizer.py --algo abc --seed $SEED --db $DB
    yazılır. Diğer her şey aynı kalır.
"""

# =========================================================
# CRITICAL: Thread limit env (TRUBA SLURM uyumluluğu)
# Tüm import'lardan ÖNCE!
# =========================================================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_NUM_THREADS", "1")
# =========================================================

import sys
import json
import argparse
import logging
import warnings
from multiprocessing import freeze_support, set_start_method

warnings.filterwarnings("ignore")

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RUN_OPTIMIZER")


def main():
    parser = argparse.ArgumentParser(
        description="Birleşik metasezgisel optimizer (VMD-TCAN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
    # ABC, seed 42 ile
    python run_optimizer.py --algo abc --seed 42 --db abc_runs/run_001/abc_running.db

    # GA, custom worker sayısı ile
    python run_optimizer.py --algo ga --seed 42 --db ga_runs/run_001/ga_running.db --workers 35

    # ABC, limit=7 (sensitivity için)
    python run_optimizer.py --algo abc --algo-params '{"limit": 7}' --seed 42 --db ...

    # Mevcut algoritma listesi
    python run_optimizer.py --list-algos

GPU politikası:
    Bu script HER ZAMAN CPU paralel çalışır. Final eğitim için ayrıca
    final_training_and_analysis.py kullanın (GPU).
        """
    )

    parser.add_argument('--algo', type=str, default=None,
                        help="Algoritma adı (abc, ga, pso, gwo, rime, cpo, ho)")
    parser.add_argument('--list-algos', action='store_true',
                        help="Mevcut algoritmaları listele ve çık")
    parser.add_argument('--mode', type=str, default='joint',
                        choices=['joint', 'sequential'],
                        help="Optimizasyon modu (default: joint)")

    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed (default: time.time())")
    parser.add_argument('--db', type=str, default=None,
                        help="SQLite veritabanı dosyası")
    parser.add_argument('--workers', type=int, default=None,
                        help="Paralel worker sayısı (default: cpu_count() - 1)")
    parser.add_argument('--runid', type=int, default=None,
                        help="Paralel çalışma ID'si (logging için)")

    parser.add_argument('--pop-size', type=int, default=40,
                        help="Popülasyon büyüklüğü (default: 40)")
    parser.add_argument('--max-iter', type=int, default=50,
                        help="Maksimum iterasyon (default: 50)")

    parser.add_argument('--algo-params', type=str, default=None,
                        help="Algoritma-spesifik parametreler (JSON string)")

    # Sequential mode — K ve alpha sabit, sadece TCAN optimize edilir (Kiyas 2)
    parser.add_argument('--fixed-wavelet', type=str, default=None,
                        choices=['db4','db5','db6','db7','db8',
                                 'sym5','sym6','sym7','sym8',
                                 'coif3','coif4','coif5'],
                        help="[Sequential] Sabit mother wavelet "
                             "(onceki paper TPE-best: 'sym5')")
    parser.add_argument('--fixed-level', type=int, default=None,
                        help="[Sequential] Sabit DWT level (onceki paper TPE-best: 6)")
    parser.add_argument('--fixed-mode', type=str, default='symmetric',
                        choices=['symmetric', 'zero', 'periodization'],
                        help="[Sequential] Sabit DWT padding mode (default: symmetric)")

    # Faz 3 ablation modlari — bilesen katkilarini izole eder
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['nodwt', 'nobidir', 'waveletfixed', 'random'],
                        help="[Ablation] Faz 4 deneyleri: bir bileseni kapat. "
                             "nodwt=DWT kapali (ham sinyal tek bilesen), "
                             "nobidir=BiLSTM yerine duz LSTM, "
                             "waveletfixed=wavelet sym5 sabit (level/mode optimize), "
                             "random=RandomSearch (ayri algoritma gerekir)")

    parser.add_argument('--analyze', action='store_true',
                        help="Optimizasyon yerine mevcut DB'yi analiz et")

    # Faz 2 walk-forward: ozel veri penceresi
    parser.add_argument('--data-window-start', type=int, default=None,
                        help="[Walk-forward] HP search icin pencere baslangic saati "
                             "(default: son OPT_DATA_LIMIT=8760 saat)")
    parser.add_argument('--data-window-end', type=int, default=None,
                        help="[Walk-forward] HP search icin pencere bitis saati. "
                             "--data-window-start ile birlikte verilmeli.")

    args = parser.parse_args()

    # --list-algos modunda erken çık
    if args.list_algos:
        from optimizers import list_algorithms
        algos = list_algorithms()
        print("Mevcut algoritmalar:")
        for a in algos:
            print(f"  - {a}")
        return 0

    # Algo zorunlu
    if not args.algo:
        parser.error("--algo argümanı zorunlu (--list-algos hariç)")

    # DB dosyası zorunlu
    if not args.db and not args.analyze:
        parser.error("--db argümanı zorunlu")

    # Analyze mode
    if args.analyze:
        from optimizers.db_manager import GenericDBManager
        db = GenericDBManager(args.db)
        summary = db.get_summary()
        print("\n" + "=" * 60)
        print(f"OPTIMIZASYON ÖZETİ — {args.db}")
        print("=" * 60)
        print(f"Total trials: {summary['total_trials']}")
        print(f"Improvements: {summary['improvements']}")
        if summary['best_loss'] is not None:
            print(f"Best loss: {summary['best_loss']:.6f}")
        print("=" * 60)
        return 0

    # Algoritma sınıfını al
    from optimizers import get_optimizer
    OptimizerClass = get_optimizer(args.algo)

    # Algo-spesifik params
    algo_params = None
    if args.algo_params:
        try:
            algo_params = json.loads(args.algo_params)
        except json.JSONDecodeError as e:
            parser.error(f"--algo-params geçersiz JSON: {e}")

    # Ablation flagleri env var'lara yaz (Faz 3); child process'e miras kalir
    if args.ablation == 'nodwt':
        os.environ["ABL_USE_DWT"] = "0"
        logger.info("[ABLATION] use_dwt=False (DWT kapali, ham sinyal)")
    elif args.ablation == 'nobidir':
        os.environ["ABL_USE_BIDIR"] = "0"
        logger.info("[ABLATION] use_bidirectional=False (BiLSTM -> duz LSTM)")
    elif args.ablation == 'waveletfixed':
        # wavelet sabit sym5 - param_mapping bu env'i okuyabilir; geçici olarak
        # mapping'in 0. boyutunu sym5'e zorla. Faz 4 koDtuktan sonra kararlastirilacak.
        os.environ["ABL_WAVELET_FIXED"] = "sym5"
        logger.info("[ABLATION] wavelet sabit 'sym5' (level/mode hala optimize)")
    elif args.ablation == 'random':
        # RandomSearch: optimizer'in update kurali her iterasyonda rastgele
        # arama yapacak sekilde davranir. Bu yontem objective.py'da degil,
        # optimizer seviyesinde uygulanmali; simdilik kullanici manuel
        # algoritma secimi yapmali (run_optimizer.py --algo random_search).
        # Su an yer tutucu olarak env var set edilmiyor.
        logger.warning("[ABLATION] random modu icin ayri 'random_search' "
                       "algoritmasi gerekiyor; bu argument NotImplemented")

    # Walk-forward: pencere env varlari (child process'e miras kalir)
    if args.data_window_start is not None and args.data_window_end is not None:
        os.environ["WF_WINDOW_START"] = str(args.data_window_start)
        os.environ["WF_WINDOW_END"]   = str(args.data_window_end)
        logger.info(
            f"[WALK-FORWARD] Veri penceresi: "
            f"[{args.data_window_start}:{args.data_window_end}] "
            f"({args.data_window_end - args.data_window_start} saat)"
        )
    elif args.data_window_start is not None or args.data_window_end is not None:
        parser.error("--data-window-start ve --data-window-end birlikte verilmeli")

    # Sequential mode: env var'lari set et (child process'e miras kalir)
    seq_dimension = None
    if args.fixed_wavelet is not None and args.fixed_level is not None:
        os.environ["SEQ_FIXED_WAVELET"] = str(args.fixed_wavelet)
        os.environ["SEQ_FIXED_LEVEL"]   = str(args.fixed_level)
        os.environ["SEQ_FIXED_MODE"]    = str(args.fixed_mode)
        from optimizers.param_mapping import DIMENSION_SEQ
        seq_dimension = DIMENSION_SEQ
        logger.info(
            f"[SEQUENTIAL] wavelet={args.fixed_wavelet}, "
            f"level={args.fixed_level}, mode={args.fixed_mode} sabit"
        )
        logger.info(f"[SEQUENTIAL] Arama boyutu: {seq_dimension}D (BiLSTM only)")
    elif args.fixed_wavelet is not None or args.fixed_level is not None:
        parser.error("--fixed-wavelet ve --fixed-level birlikte verilmeli")

    logger.info("=" * 60)
    logger.info(f"OPTIMIZER: {args.algo.upper()}")
    logger.info("=" * 60)
    logger.info(f"DB: {args.db}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Pop: {args.pop_size}, MaxIter: {args.max_iter}")
    if algo_params:
        logger.info(f"Algo params override: {algo_params}")

    # Optimizer'ı oluştur ve çalıştır
    optimizer_kwargs = dict(
        db_file=args.db,
        seed=args.seed,
        runid=args.runid,
        pop_size=args.pop_size,
        max_iterations=args.max_iter,
        n_workers=args.workers,
        algo_params=algo_params,
        mode=args.mode,
    )
    if seq_dimension is not None:
        optimizer_kwargs['dimension'] = seq_dimension

    optimizer = OptimizerClass(**optimizer_kwargs)

    try:
        best_vector, best_loss = optimizer.run()

        # En iyi parametreleri JSON olarak kaydet
        from optimizers.param_mapping import (
            map_vector_to_params, map_vector_to_params_seq, params_to_str_keys
        )
        if seq_dimension is not None:
            best_params = map_vector_to_params_seq(
                best_vector,
                fixed_wavelet=args.fixed_wavelet,
                fixed_level=int(args.fixed_level),
                fixed_mode=args.fixed_mode,
            )
        else:
            best_params = map_vector_to_params(best_vector)
        save_data = params_to_str_keys(best_params)
        save_data['_loss'] = float(best_loss)
        save_data['_seed'] = optimizer.seed
        save_data['_algorithm'] = args.algo
        save_data['_mode'] = args.mode
        if seq_dimension is not None:
            save_data['_fixed_wavelet'] = args.fixed_wavelet
            save_data['_fixed_level']   = args.fixed_level
            save_data['_fixed_mode']    = args.fixed_mode
        if args.data_window_start is not None:
            save_data['_data_window_start'] = args.data_window_start
            save_data['_data_window_end']   = args.data_window_end

        json_file = args.db.replace('.db', '_best_params.json')
        if json_file == args.db:
            json_file = f"{args.algo}_best_params.json"

        with open(json_file, "w") as f:
            json.dump(save_data, f, indent=4)

        logger.info(f"\nEn iyi parametreler kaydedildi: {json_file}")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nKullanici durdurdu. Durum DB'ye kaydedildi.")
        logger.warning("Ayni komutla tekrar calistirirsan kaldigi yerden devam eder.")
        return 130


if __name__ == '__main__':
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    freeze_support()
    sys.exit(main())
