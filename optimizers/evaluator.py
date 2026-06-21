# -*- coding: utf-8 -*-
"""
optimizers/evaluator.py — Paralel fitness değerlendirme (DWT+BiLSTM, v8).

Tüm algoritmalar için ortak worker fonksiyonu. multiprocessing.Pool ile
çağrılır; her worker kendi WindForecastObjective örneğini oluşturur ve
bir 9D vektörü değerlendirip sonucu döndürür.

Önemli: Worker'lar HER ZAMAN CPU üzerinde çalışır (use_gpu=False).
Paralel worker'lar GPU bellek çakışması yaşar; GPU sadece final eğitimde
kullanılır.

Mimari (v9): DWT + BiLSTM. Önceki TCAN+VMD evaluator'dan değişiklikler:
    - 10D -> 9D vektör (K, ALPHA, KERNEL, ATT_DIM kaldirildi)
    - Ablation flagleri: ABL_USE_DWT (yerine ABL_USE_VMD), ABL_USE_BIDIR
      (yerine ABL_USE_ATTENTION). PADP/ABL_USE_PENALTY tamamen kaldirildi
      (v9, Faz 3 ablation negligible cikti).
    - Sequential mode env varlari: SEQ_FIXED_WAVELET, SEQ_FIXED_LEVEL,
      SEQ_FIXED_MODE (yerine SEQ_FIXED_K, SEQ_FIXED_ALPHA)
"""

import os
import time
from datetime import datetime

# Thread limit env variables — import'lardan önce ayarlanmalı (TRUBA için)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_NUM_THREADS", "1")

from objective import WindForecastObjective
from utils import load_ruzgar_verisi
from optimizers.param_mapping import (
    map_vector_to_params, map_vector_to_params_seq, calculate_fitness
)


def init_worker():
    """Worker process başlatma — veriyi belleğe yükler.

    multiprocessing.Pool'a `initializer=init_worker` olarak verilir.
    """
    try:
        load_ruzgar_verisi()
    except Exception as e:
        print(f"[Worker init] Veri yükleme hatası: {e}")


def evaluate_solution_wrapper(args):
    """Tek bir vektörü değerlendiren worker fonksiyonu.

    Args:
        args: tuple — (vector, member_id, member_type, iteration, runid, mode)
            vector: numpy array, [0,1]^9 (joint) veya [0,1]^6 (sequential)
            member_id: int — popülasyondaki birey indeksi
            member_type: str — "I" (init), "E"/"O"/"S" (ABC), "P"/"C" (GA), vb.
            iteration: int — cycle/generation numarası
            runid: int veya None — paralel çalışma ID'si
            mode: str — "joint" veya "sequential"

    Returns:
        dict: {
            'loss': float,
            'fitness': float,
            'params': dict (HP enum -> value),
            'params_str': dict (str -> value, DB için),
            'vector': list,
            'elapsed_seconds': float,
            'member_id': int,
            'member_type': str,
            'iteration': int,
            'timestamp': str (ISO format),
            'component_losses': list (PADP diagnostic için, opsiyonel)
        }
    """
    vector, member_id, member_type, iteration, runid, mode = args

    start_time = time.time()

    # Ablation flagleri env var'lardan oku (Faz 4).
    # run_optimizer.py --ablation X argumaniyla set edilir; child process'e
    # miras kalir (spawn/fork ile multiprocessing.Pool'e gecer).
    _use_dwt   = os.environ.get("ABL_USE_DWT", "1") != "0"
    _use_bidir = os.environ.get("ABL_USE_BIDIR", "1") != "0"

    # Walk-forward (Faz 2): WF_WINDOW_START / WF_WINDOW_END env varlari
    # walk_forward_eval.py tarafindan set edilir. Yoksa default davranis.
    _wf_start = os.environ.get("WF_WINDOW_START")
    _wf_end   = os.environ.get("WF_WINDOW_END")
    _data_window = None
    if _wf_start is not None and _wf_end is not None:
        _data_window = (int(_wf_start), int(_wf_end))

    # CPU zorunlu — paralel worker'lar GPU paylaşamaz
    obj = WindForecastObjective(
        mode="global",
        is_final_training=False,
        show=0,
        use_gpu=False,
        use_dwt=_use_dwt,
        use_bidirectional=_use_bidir,
        data_window=_data_window,
    )

    # Sequential mode env varlari: prepare scripti tarafindan set edilir.
    # Yoksa joint mode (9D) kullanilir.
    _fixed_wavelet = os.environ.get("SEQ_FIXED_WAVELET")
    _fixed_level   = os.environ.get("SEQ_FIXED_LEVEL")
    _fixed_mode    = os.environ.get("SEQ_FIXED_MODE", "symmetric")

    if _fixed_wavelet is not None and _fixed_level is not None:
        # Sequential: 6D vector, DWT params sabit
        params = map_vector_to_params_seq(
            vector,
            fixed_wavelet=_fixed_wavelet,
            fixed_level=int(_fixed_level),
            fixed_mode=_fixed_mode,
        )
    else:
        # Joint: 9D vector
        params = map_vector_to_params(vector)

    runid_str = "T" if runid is None else str(runid)
    log_prefix = f" i:{runid_str}-{iteration} | ind {member_id} ({member_type}) |"

    # Objective cagrisi: dict[HP, value] -> loss.
    # objective.py opsiyonel olarak (loss, component_losses) dondurebilir
    # (PADP diagnostic icin).
    component_losses = None
    try:
        result = obj(params, log_prefix)
        if isinstance(result, tuple):
            loss, component_losses = result
        else:
            loss = result
    except Exception as e:
        print(f"[Worker error] {e}")
        loss = float('inf')

    elapsed = time.time() - start_time

    out = {
        'loss': float(loss),
        'fitness': calculate_fitness(loss),
        'params': params,
        'params_str': {str(k.value): v for k, v in params.items()},
        'vector': vector.tolist() if hasattr(vector, 'tolist') else list(vector),
        'elapsed_seconds': elapsed,
        'member_id': member_id,
        'member_type': member_type,
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
    }
    if component_losses is not None:
        out['component_losses'] = list(component_losses)
    return out
