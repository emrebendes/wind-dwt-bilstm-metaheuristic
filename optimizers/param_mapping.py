# -*- coding: utf-8 -*-
"""
optimizers/param_mapping.py — [0,1] vektörü → DWT+BiLSTM hiperparametreleri.

9 boyutlu birim hiperküp [0,1]^9 üzerindeki vektörleri DWT+BiLSTM
hiperparametrelerine dönüştürür. Tüm metasezgisel algoritmalar bu
ortak mapping'i kullanır — adil karşılaştırma için kritik.

Aynı vektör → aynı parametreler garantisi vardır (deterministic mapping).

Mimari (v8, 2026-05-28):
    - Önceki paper'da (Manuscript.docx) H=1 için en iyi: DWT+BiLSTM
    - Önceki paper TPE-best: wavelet='sym5', level=6
    - Bu paper: 8 metasezgisel ile aynı 9D arama uzayinda joint optimization

Boyutlar (9D):
    0 - DWT_WAVELET (kategorik, 12 secenek; utils.WAVELET_CHOICES)
    1 - DWT_LEVEL   (int, 3-7)
    2 - DWT_MODE    (kategorik, 3 secenek; utils.MODE_CHOICES)
    3 - LOOK_BACK   (int, 24-96)
    4 - HIDDEN      (int, 32-256)
    5 - LAYERS      (int, 1-4)
    6 - DROPOUT     (float, 0.0-0.5; linear)
    7 - LR          (float, 1e-4 - 1e-2; LOG-uniform)
    8 - BATCH       (kategorik, {32, 64, 128})
"""

import numpy as np

from utils import HP, WAVELET_CHOICES, MODE_CHOICES, BATCH_CHOICES


# =============================================================================
# PARAMETRE SINIRLARI (BOUNDS)
# Tüm algoritmalar bu sınırları kullanır.
# =============================================================================

BOUNDS = {
    # DWT
    HP.DWT_WAVELET: (0, len(WAVELET_CHOICES) - 1e-9),  # kategorik index
    HP.DWT_LEVEL:   (3, 7),                            # int
    HP.DWT_MODE:    (0, len(MODE_CHOICES) - 1e-9),     # kategorik index
    # Data
    HP.LOOK_BACK:   (24, 96),                          # int
    # BiLSTM
    HP.HIDDEN:      (32, 256),                         # int
    HP.LAYERS:      (1, 4),                            # int
    HP.DROPOUT:     (0.0, 0.5),                        # float (linear)
    # Training
    HP.LR:          (1e-4, 1e-2),                      # float (LOG)
    HP.BATCH:       (0, len(BATCH_CHOICES) - 1e-9),    # kategorik index
}

DIMENSION = 9  # Joint optimization toplam boyut

# Vektör indeksi → HP (sıralı, deterministic)
_VECTOR_INDEX_TO_HP = [
    HP.DWT_WAVELET,  # 0  kategorik
    HP.DWT_LEVEL,    # 1  int
    HP.DWT_MODE,     # 2  kategorik
    HP.LOOK_BACK,    # 3  int
    HP.HIDDEN,       # 4  int
    HP.LAYERS,       # 5  int
    HP.DROPOUT,      # 6  float linear
    HP.LR,           # 7  float LOG
    HP.BATCH,        # 8  kategorik
]


# =============================================================================
# YARDIMCI MAP FONKSIYONLARI (tip-spesifik)
# =============================================================================

def _map_categorical(u, choices):
    """[0,1] -> kategori değerinden birini secer.

    Kategori sayisina esit aralikla bolar; ust sinirda taSma onlenir.
    """
    n = len(choices)
    idx = min(int(u * n), n - 1)
    return choices[idx]


def _map_int_linear(u, lo, hi):
    """[0,1] -> int [lo, hi] (her iki uc dahil).

    np.round kullanilir; lo ve hi'in inclusive olmasi icin (hi - lo + 1)
    bolme aralikla esit dagilim saglar.
    """
    n = int(hi - lo + 1)
    idx = min(int(u * n), n - 1)
    return int(lo + idx)


def _map_float_linear(u, lo, hi):
    """[0,1] -> float [lo, hi] (linear)."""
    return float(u * (hi - lo) + lo)


def _map_float_log(u, lo, hi):
    """[0,1] -> float [lo, hi] (log-uniform).

    LR gibi log-scale parametreler icin: u=0 -> lo, u=1 -> hi,
    fakat aradaki dagilim logaritmik.
    """
    log_lo = np.log(lo)
    log_hi = np.log(hi)
    return float(np.exp(u * (log_hi - log_lo) + log_lo))


# =============================================================================
# JOINT MODE — 9D mapping
# =============================================================================

def map_vector_to_params(vector):
    """[0,1]^9 vektörünü gerçek DWT+BiLSTM hiperparametrelerine çevirir.

    Args:
        vector: numpy array veya list, 9 elemanli, her elemani [0,1] araliginda

    Returns:
        dict[HP, value]:
            {HP.DWT_WAVELET: 'sym5', HP.DWT_LEVEL: 6, ..., HP.BATCH: 64}
    """
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape[-1] != DIMENSION:
        raise ValueError(
            f"Vektör boyutu {vector.shape[-1]}, beklenen {DIMENSION}"
        )

    params = {}

    # 0: DWT_WAVELET (kategorik)
    params[HP.DWT_WAVELET] = _map_categorical(vector[0], WAVELET_CHOICES)

    # 1: DWT_LEVEL (int 3-7)
    params[HP.DWT_LEVEL] = _map_int_linear(vector[1], 3, 7)

    # 2: DWT_MODE (kategorik)
    params[HP.DWT_MODE] = _map_categorical(vector[2], MODE_CHOICES)

    # 3: LOOK_BACK (int 24-96)
    params[HP.LOOK_BACK] = _map_int_linear(vector[3], 24, 96)

    # 4: HIDDEN (int 32-256)
    params[HP.HIDDEN] = _map_int_linear(vector[4], 32, 256)

    # 5: LAYERS (int 1-4)
    params[HP.LAYERS] = _map_int_linear(vector[5], 1, 4)

    # 6: DROPOUT (float linear 0.0-0.5)
    params[HP.DROPOUT] = _map_float_linear(vector[6], 0.0, 0.5)

    # 7: LR (float LOG 1e-4 - 1e-2)
    params[HP.LR] = _map_float_log(vector[7], 1e-4, 1e-2)

    # 8: BATCH (kategorik)
    params[HP.BATCH] = _map_categorical(vector[8], BATCH_CHOICES)

    return params


# =============================================================================
# SEQUENTIAL MODE — 6D mapping (DWT params sabit)
# =============================================================================

DIMENSION_SEQ = 6  # BiLSTM-only: LOOK_BACK, HIDDEN, LAYERS, DROPOUT, LR, BATCH

_SEQ_VECTOR_INDEX_TO_HP = [
    HP.LOOK_BACK,  # 0  int
    HP.HIDDEN,     # 1  int
    HP.LAYERS,     # 2  int
    HP.DROPOUT,    # 3  float linear
    HP.LR,         # 4  float LOG
    HP.BATCH,      # 5  kategorik
]


def map_vector_to_params_seq(vector_6d,
                              fixed_wavelet: str = 'sym5',
                              fixed_level: int = 6,
                              fixed_mode: str = 'symmetric') -> dict:
    """6D [0,1] vektörünü BiLSTM hyperparametrelerine + sabit DWT param'larina dönüstür.

    Sequential modda:
      - DWT parametreleri envelope-entropy/onceki paper TPE-best ile sabit
        (default: wavelet='sym5', level=6, mode='symmetric')
      - Sadece BiLSTM (4) + training (2) = 6 boyut optimize edilir
      - Adil compute: joint ile ayni budget, 6D vs 9D fark sadece arama uzayinda

    Args:
        vector_6d: 6 elemanli [0,1] araliginda numpy array
        fixed_wavelet: sabit mother wavelet (onceki paper TPE-best: 'sym5')
        fixed_level: sabit DWT level (onceki paper TPE-best: 6)
        fixed_mode: sabit padding modu ('symmetric' default)

    Returns:
        dict[HP, value]: 9 anahtarli tam dict (DWT'ler sabit, BiLSTM optimize)
    """
    v = np.asarray(vector_6d, dtype=np.float64)
    if len(v) != DIMENSION_SEQ:
        raise ValueError(
            f"Beklenen {DIMENSION_SEQ}D, gelen {len(v)}D"
        )

    if fixed_wavelet not in WAVELET_CHOICES:
        raise ValueError(
            f"fixed_wavelet '{fixed_wavelet}' WAVELET_CHOICES icinde degil"
        )
    if fixed_mode not in MODE_CHOICES:
        raise ValueError(
            f"fixed_mode '{fixed_mode}' MODE_CHOICES icinde degil"
        )

    params = {}

    # Sabit DWT parametreleri
    params[HP.DWT_WAVELET] = fixed_wavelet
    params[HP.DWT_LEVEL]   = int(fixed_level)
    params[HP.DWT_MODE]    = fixed_mode

    # Optimize edilen BiLSTM + training (6 boyut)
    params[HP.LOOK_BACK] = _map_int_linear(v[0], 24, 96)
    params[HP.HIDDEN]    = _map_int_linear(v[1], 32, 256)
    params[HP.LAYERS]    = _map_int_linear(v[2], 1, 4)
    params[HP.DROPOUT]   = _map_float_linear(v[3], 0.0, 0.5)
    params[HP.LR]        = _map_float_log(v[4], 1e-4, 1e-2)
    params[HP.BATCH]     = _map_categorical(v[5], BATCH_CHOICES)

    return params


# =============================================================================
# YARDIMCILAR
# =============================================================================

def params_to_str_keys(params: dict) -> dict:
    """HP enum anahtarlarini string'e cevir (DB / JSON yazimi icin)."""
    return {str(k.value): v for k, v in params.items()}


def calculate_fitness(loss: float) -> float:
    """Standart fitness donusumu (ABC ve diger algoritmalar icin).

    Dusuk loss -> yuksek fitness.
    """
    if loss >= 0:
        return 1.0 / (loss + 1.0)
    return 1.0 + abs(loss)


def params_to_vector(params: dict) -> np.ndarray:
    """Ters mapping: HP dict -> [0,1]^9 vektör.

    Çoğunlukla testing/debugging ve "best params'ı vektör olarak kaydet"
    kullanım durumları için. Kategori değerler için merkez index kullanılır.

    NOT: int parametreler için en yakın grid noktasına yuvarlama yapilir,
    yani round-trip exact olmayabilir (off-by-one bin sapması olabilir).
    """
    v = np.zeros(DIMENSION, dtype=np.float64)

    # 0: DWT_WAVELET (kategori indeksi -> ortalanmis [0,1])
    w_idx = WAVELET_CHOICES.index(params[HP.DWT_WAVELET])
    v[0] = (w_idx + 0.5) / len(WAVELET_CHOICES)

    # 1: DWT_LEVEL (int 3-7 -> [0,1])
    n_levels = 7 - 3 + 1
    v[1] = (int(params[HP.DWT_LEVEL]) - 3 + 0.5) / n_levels

    # 2: DWT_MODE (kategori)
    m_idx = MODE_CHOICES.index(params[HP.DWT_MODE])
    v[2] = (m_idx + 0.5) / len(MODE_CHOICES)

    # 3: LOOK_BACK (int 24-96)
    n_lb = 96 - 24 + 1
    v[3] = (int(params[HP.LOOK_BACK]) - 24 + 0.5) / n_lb

    # 4: HIDDEN (int 32-256)
    n_h = 256 - 32 + 1
    v[4] = (int(params[HP.HIDDEN]) - 32 + 0.5) / n_h

    # 5: LAYERS (int 1-4)
    n_l = 4 - 1 + 1
    v[5] = (int(params[HP.LAYERS]) - 1 + 0.5) / n_l

    # 6: DROPOUT (float linear)
    v[6] = (float(params[HP.DROPOUT]) - 0.0) / (0.5 - 0.0)

    # 7: LR (float LOG)
    log_lo, log_hi = np.log(1e-4), np.log(1e-2)
    v[7] = (np.log(float(params[HP.LR])) - log_lo) / (log_hi - log_lo)

    # 8: BATCH (kategori)
    b_idx = BATCH_CHOICES.index(int(params[HP.BATCH]))
    v[8] = (b_idx + 0.5) / len(BATCH_CHOICES)

    return np.clip(v, 0.0, 1.0)
