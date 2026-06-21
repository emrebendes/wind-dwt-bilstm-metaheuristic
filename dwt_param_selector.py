#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dwt_param_selector.py — Faz 3 Sequential Stage 1.

Amac: Tum (wavelet, level, mode) kombosu uzerinde Shannon entropy hesabi
yaparak en kompakt DWT temsilini secer. Bu, sequential optimization'in 1.
asamasidir: secilen wavelet/level/mode SABIT tutulur, sonra 6D BiLSTM HP'leri
metasezgisel ile optimize edilir.

Bilimsel ref:
    Sang Y.-F. (2009) "A practical guide to discrete wavelet decomposition of
    hydrologic time series" J. Hydrology — su zaman serileri icin DWT wavelet
    seciminde entropy kriteri standart.
    Wickerhauser M.V. (1994) "Adapted Wavelet Analysis from Theory to Software"
    — best-basis selection.

Yaklasim:
    - 12 wavelet × 5 level (3-7) × 3 mode = 180 kombinasyon
    - Her kombinasyon icin 8 istasyonda DWT detail coefficients Shannon entropy
    - Toplam entropy minimum olan kombo seçilir
    - Sonuc: dwt_selection_result.json
    - Sequential optimization (run_optimizer.py --fixed-wavelet X --fixed-level Y
      --fixed-mode Z) bu sonucu kullanir.

Kullanim:
    python dwt_param_selector.py
    # Cikti: dwt_selection_result.json
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime

import numpy as np
import pywt

# Thread limit
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

from utils import load_ruzgar_verisi, WAVELET_CHOICES, MODE_CHOICES
from config import OPT_DATA_LIMIT

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [DWT_SEL] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DWT_SEL")


LEVELS = [3, 4, 5, 6, 7]


def shannon_entropy(coeffs: np.ndarray) -> float:
    """Tek bir detail coefficient dizisinin Shannon entropy'sini hesapla.

    H = -sum(p_i * log(p_i))
    p_i = normalized squared coefficients.
    """
    sq = np.asarray(coeffs, dtype=np.float64) ** 2
    total = sq.sum()
    if total <= 1e-12:
        return 0.0
    p = sq / total
    # log(0) = -inf, p*log(0) = 0 — np.where ile guvenli
    p_safe = np.where(p > 0, p, 1.0)
    return float(-(p * np.log(p_safe)).sum())


def total_entropy_for_signal(signal: np.ndarray, wavelet: str,
                             level: int, mode: str) -> float:
    """Bir sinyal icin DWT yapip tum bilesenlerin toplam Shannon entropy'si."""
    try:
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level, mode=mode)
        # coeffs = [cA_n, cD_n, cD_n-1, ..., cD_1]
        # Tum bilesenlerin Shannon entropy toplami
        total = sum(shannon_entropy(c) for c in coeffs)
        return total
    except Exception as e:
        # Bazi wavelet+level kombolari saglanmaz; inf doner -> bu kombo elenir
        return float('inf')


def select_best_dwt_params(data_window: tuple = None) -> dict:
    """Tum (wavelet, level, mode) kombolarini tarayip en iyiyi sec.

    Args:
        data_window: (start, end) hour indices. None ise son OPT_DATA_LIMIT
                     saat (Faz 1 ile uyumlu).

    Returns:
        dict: best params + tum grid sonuclari.
    """
    logger.info("Veri yukleniyor...")
    raw_data = load_ruzgar_verisi()
    logger.info(f"Veri shape: {raw_data.shape}")

    # Pencere se
    signals = []
    if data_window is not None:
        start, end = data_window
        for s in raw_data:
            signals.append(np.array(s[start:end], dtype=np.float64))
        win_label = f"hour [{start}:{end}]"
    else:
        for s in raw_data:
            sig = np.array(s, dtype=np.float64)
            if len(sig) > OPT_DATA_LIMIT:
                sig = sig[-OPT_DATA_LIMIT:]
            signals.append(sig)
        win_label = f"son {OPT_DATA_LIMIT} saat (Faz 1 uyumlu)"

    logger.info(f"Veri penceresi: {win_label}, istasyon sayisi: {len(signals)}, "
                f"sinyal uzunlugu: {len(signals[0])}")

    # Grid search
    n_combos = len(WAVELET_CHOICES) * len(LEVELS) * len(MODE_CHOICES)
    logger.info(f"Grid arama baslatildi: {n_combos} kombo "
                f"({len(WAVELET_CHOICES)} wavelet × {len(LEVELS)} level × "
                f"{len(MODE_CHOICES)} mode)")

    results = []
    t0 = time.time()
    for wavelet in WAVELET_CHOICES:
        for level in LEVELS:
            for mode in MODE_CHOICES:
                # 8 istasyon toplam entropy'si
                total_ent = 0.0
                valid = True
                for sig in signals:
                    ent = total_entropy_for_signal(sig, wavelet, level, mode)
                    if not np.isfinite(ent):
                        valid = False
                        break
                    total_ent += ent
                if not valid:
                    continue
                results.append({
                    "wavelet": wavelet,
                    "level": level,
                    "mode": mode,
                    "total_entropy": total_ent,
                })

    elapsed = time.time() - t0
    if not results:
        logger.error("Hicbir gecerli kombo bulunamadi!")
        return {}

    # En dusuk total_entropy
    results.sort(key=lambda r: r["total_entropy"])
    best = results[0]

    logger.info(f"\nGrid arama tamamlandi ({elapsed:.1f} s, {len(results)} gecerli kombo)")
    logger.info(f"En iyi (min entropy): wavelet={best['wavelet']}, "
                f"level={best['level']}, mode={best['mode']}, "
                f"H_total={best['total_entropy']:.4f}")
    logger.info(f"\nTop 5 kombo:")
    for i, r in enumerate(results[:5]):
        logger.info(f"  {i+1}. wavelet={r['wavelet']:6s}, level={r['level']}, "
                    f"mode={r['mode']:14s} H={r['total_entropy']:.4f}")

    return {
        "selected_wavelet": best["wavelet"],
        "selected_level": best["level"],
        "selected_mode": best["mode"],
        "min_total_entropy": best["total_entropy"],
        "n_stations": len(signals),
        "signal_length": int(len(signals[0])),
        "data_window": list(data_window) if data_window else None,
        "n_combos_tested": len(results),
        "elapsed_seconds": elapsed,
        "method": "Shannon entropy of DWT detail coefficients (per-station sum)",
        "wavelet_choices": list(WAVELET_CHOICES),
        "level_choices": LEVELS,
        "mode_choices": list(MODE_CHOICES),
        "grid_results": results,   # tum sonuclar (top entropy artan sirada)
        "created_at": datetime.now().isoformat(),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="DWT params (wavelet, level, mode) Shannon entropy ile sec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", type=str,
                        default="dwt_selection_result.json",
                        help="Cikti JSON yolu (default: dwt_selection_result.json)")
    parser.add_argument("--data-window-start", type=int, default=None,
                        help="Pencere baslangic saati (opsiyonel)")
    parser.add_argument("--data-window-end", type=int, default=None,
                        help="Pencere bitis saati (opsiyonel)")
    args = parser.parse_args()

    data_window = None
    if args.data_window_start is not None and args.data_window_end is not None:
        data_window = (args.data_window_start, args.data_window_end)

    result = select_best_dwt_params(data_window=data_window)
    if not result:
        return 1

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nKaydedildi: {args.output}")

    logger.info(f"\n=== SEQUENTIAL STAGE 2 KOMUTU (paper'a koy) ===")
    logger.info(f"python run_optimizer.py --algo <ALGO> --seed <SEED> --db <DB> \\")
    logger.info(f"    --fixed-wavelet {result['selected_wavelet']} \\")
    logger.info(f"    --fixed-level {result['selected_level']} \\")
    logger.info(f"    --fixed-mode {result['selected_mode']} \\")
    logger.info(f"    --pop-size 40 --max-iter 50 --workers 35")
    return 0


if __name__ == "__main__":
    sys.exit(main())
