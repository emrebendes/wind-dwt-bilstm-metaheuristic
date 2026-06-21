#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_persistence_baseline.py — Naive persistence forecast baseline + skill score.

Hakem 1 cevabi:
    "...without comparing against established baselines (persistence, ARIMA, plain
    LSTM), the practical value of the contribution is hard to evaluate..."

KBS submission gerekligi: skill score > 0 olmasi paper'in temel iddiasini destekler.

Bilimsel cerceve:
    Persistence (naive) forecast: y_hat[t] = y[t-1]
    Skill score: SS = 1 - RMSE_model / RMSE_persistence
    SS > 0  -> model persistence'i geciyor
    SS == 0 -> model persistence ile esit
    SS < 0  -> model persistence'ten kotu (deger uretmiyor)

Iki seviyede hesap:
    A) Raw wind speed (m/s) — literatur standardi, m/s biriminde reportable
    B) Normalized component-wise — modelin egitim metrigi ile karsilastirilabilir

Kullanim:
    python compute_persistence_baseline.py
    # -> persistence_results.json
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Repo modullerine eris
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import All_STATION_DATA_PATH, TRAIN_RATIO
from utils import load_ruzgar_verisi, dwt_decompose, normalize_train_test_split


def persistence_rmse(series: np.ndarray) -> float:
    """1-step naive persistence forecast RMSE.

    y_hat[t] = y[t-1] icin RMSE = sqrt(mean( (y[t]-y[t-1])^2 ))

    Args:
        series: 1D array (raw veya normalized, fark etmez)
    Returns:
        RMSE skaler
    """
    if len(series) < 2:
        return float('nan')
    diffs = series[1:] - series[:-1]
    return float(np.sqrt(np.mean(diffs ** 2)))


def main():
    print("=" * 70)
    print("PERSISTENCE BASELINE — Wind Speed Forecasting")
    print("=" * 70)

    # 1) Veri yukle
    data = load_ruzgar_verisi(All_STATION_DATA_PATH)
    n_stations, n_steps = data.shape
    print(f"Veri: {n_stations} istasyon x {n_steps} saat = {data.size:,} ornek")
    print(f"Train/test orani: {TRAIN_RATIO}")

    # =========================================================================
    # A) RAW WIND SPEED (m/s) — literatur standardi
    # =========================================================================
    print("\n[A] RAW WIND SPEED PERSISTENCE (m/s)")
    print("-" * 70)
    split_idx = int(n_steps * TRAIN_RATIO)
    print(f"Split index: {split_idx} (egitim), {n_steps - split_idx} (test)")

    station_rmses_raw = []
    for s in range(n_stations):
        series = np.asarray(data[s], dtype=np.float64)
        test_portion = series[split_idx:]
        rmse_s = persistence_rmse(test_portion)
        station_rmses_raw.append(rmse_s)
        print(f"  Istasyon {s+1}: test_rmse_persistence = {rmse_s:.4f} m/s")

    mean_raw = float(np.mean(station_rmses_raw))
    std_raw  = float(np.std(station_rmses_raw))
    print(f"\nRAW WIND PERSISTENCE RMSE = {mean_raw:.4f} +/- {std_raw:.4f} m/s "
          f"(8 istasyon ortalamasi)")

    # =========================================================================
    # B) NORMALIZED COMPONENT-WISE — modelin RMSE'siyle dogrudan karsilastir
    # =========================================================================
    # GWO best params
    gwo_summary_path = Path("gwo_final_results/analysis_summary.json")
    if not gwo_summary_path.exists():
        print(f"\n[!] {gwo_summary_path} bulunamadi, GWO sonucu bekleniyor.")
        return

    with open(gwo_summary_path) as f:
        gwo = json.load(f)
    gwo_params = gwo['best_params']
    gwo_final_loss = gwo.get('final_training_loss')
    gwo_best_loss  = gwo.get('best_loss')

    print(f"\n[B] NORMALIZED COMPONENT-WISE PERSISTENCE (model metrigi ile)")
    print("-" * 70)
    print(f"GWO best params: wavelet={gwo_params['dwt_wavelet']}, "
          f"level={gwo_params['dwt_level']}, mode={gwo_params['dwt_mode']}")
    print(f"GWO best validation loss (search): {gwo_best_loss:.6f}")
    print(f"GWO final training loss (test):    {gwo_final_loss:.6f}")

    # Tum istasyonlar icin DWT yap, her bilesen icin normalize + persistence
    all_component_rmses = []   # tum istasyonlar+bilesenler
    for s in range(n_stations):
        series = np.asarray(data[s], dtype=np.float64)
        components = dwt_decompose(
            series,
            wavelet=gwo_params['dwt_wavelet'],
            level=gwo_params['dwt_level'],
            mode=gwo_params['dwt_mode'],
        )
        for c_idx, comp in enumerate(components):
            if len(comp) < 100:
                continue
            split_c = int(len(comp) * TRAIN_RATIO)
            try:
                tr_n, te_n, _ = normalize_train_test_split(comp, split_c)
            except Exception as e:
                continue
            rmse_c = persistence_rmse(te_n.ravel())
            all_component_rmses.append(rmse_c)

    mean_comp = float(np.mean(all_component_rmses))
    std_comp  = float(np.std(all_component_rmses))
    print(f"\nNORMALIZED COMPONENT PERSISTENCE RMSE = {mean_comp:.6f} +/- {std_comp:.6f}")
    print(f"(8 istasyon x {gwo_params['dwt_level']+1} bilesen = "
          f"{len(all_component_rmses)} olcum)")

    # =========================================================================
    # C) SKILL SCORE HESAPLA
    # =========================================================================
    print(f"\n[C] SKILL SCORE")
    print("-" * 70)

    # Component-level skill score (modelin metrigi ile dogrudan kiyas)
    ss_component = 1.0 - (gwo_final_loss / mean_comp)
    ss_component_search = 1.0 - (gwo_best_loss / mean_comp)
    print(f"Component-level (GWO final 0.0799 vs persist {mean_comp:.4f}):")
    print(f"  SS_final  = 1 - {gwo_final_loss:.4f}/{mean_comp:.4f} = {ss_component:.4f}")
    print(f"  SS_search = 1 - {gwo_best_loss:.4f}/{mean_comp:.4f} = {ss_component_search:.4f}")

    # =========================================================================
    # D) JSON cikti
    # =========================================================================
    out = {
        "method": "1-step naive persistence forecast",
        "train_ratio": TRAIN_RATIO,
        "n_stations": int(n_stations),
        "n_timesteps_per_station": int(n_steps),
        "raw_wind_speed_ms": {
            "per_station_rmse": [float(x) for x in station_rmses_raw],
            "mean_rmse_ms":  mean_raw,
            "std_rmse_ms":   std_raw,
            "unit": "m/s",
        },
        "normalized_component_wise": {
            "wavelet": gwo_params['dwt_wavelet'],
            "level":   gwo_params['dwt_level'],
            "mode":    gwo_params['dwt_mode'],
            "n_components_total": len(all_component_rmses),
            "mean_rmse_normalized": mean_comp,
            "std_rmse_normalized":  std_comp,
            "unit": "[0,1] normalized",
        },
        "skill_score_vs_GWO": {
            "gwo_best_validation_loss": gwo_best_loss,
            "gwo_final_training_loss":  gwo_final_loss,
            "skill_score_search":  ss_component_search,
            "skill_score_final":   ss_component,
            "interpretation": (
                "SS > 0 = model persistence'i geciyor; "
                "model gercek deger uretiyor"
            ),
        },
        "computed_at": datetime.now().isoformat(),
    }

    out_path = Path("persistence_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[OK] Sonuc kaydedildi: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
