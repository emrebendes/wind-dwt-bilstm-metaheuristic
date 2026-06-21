# -*- coding: utf-8 -*-
"""
utils.py - DWT+BiLSTM joint optimization yardimci fonksiyonlari.

Mimari (v7, 2026-05-28):
    - Dekompozisyon: Discrete Wavelet Transform (PyWavelets pywt.wavedec)
    - Model: Bidirectional LSTM (torch.nn.LSTM, bidirectional=True)
    - Onceki paper'da H=1 icin optimal kombinasyon (RMSE=0.0357, R2=0.9994)

Onceki TCAN+VMD mimarisinden yapilan degisiklikler bu dosyada:
    - HP enum: K, ALPHA, KERNEL, ATT_DIM kaldirildi
    - HP enum: DWT_WAVELET, DWT_LEVEL, DWT_MODE eklendi
    - suggest_vmd_*_params() yerine suggest_dwt_bilstm_params()
    - load_ruzgar_verisi, setup_logger, create_dataset, normalize_train_test_split
      mimari-agnostic oldugu icin korundu
"""

import os
import sys
import logging
from enum import Enum

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import All_STATION_DATA_PATH


# =============================================================================
# 1. VERI YUKLEME (CACHE)
# =============================================================================

_RUZGAR_CACHE = None


def load_ruzgar_verisi(path=All_STATION_DATA_PATH):
    """8 istasyonun rüzgar hizi verisini yukler (cache'lenir).

    Returns:
        numpy.ndarray: shape (n_stations, n_timesteps), m/s cinsinden
    """
    global _RUZGAR_CACHE

    if _RUZGAR_CACHE is not None:
        return _RUZGAR_CACHE

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} bulunamadi. Once preprocess_excel.py calistirilmali."
        )

    _RUZGAR_CACHE = np.load(path, allow_pickle=True)
    return _RUZGAR_CACHE


# =============================================================================
# 2. LOGGER
# =============================================================================

def setup_logger(name="DWT_BILSTM", level=logging.INFO):
    """Tek merkezli logger kurulumu."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# =============================================================================
# 3. HIPERPARAMETRE ENUM (DWT + BiLSTM, 9D arama uzayi)
# =============================================================================

class HP(str, Enum):
    """DWT + BiLSTM ortak hiperparametre enum.

    Optimize edilen 9 boyut:
        1- DWT_WAVELET    : kategori, mother wavelet (db4-8, sym5-8, coif3-5; 12 secenek)
                            Onceki paper TPE-best: 'sym5'
        2- DWT_LEVEL      : int, ayriStirma seviyesi (3-7; onceki paper araligi)
                            Onceki paper TPE-best: 6
        3- DWT_MODE       : kategori, padding ('symmetric', 'zero', 'periodization')
        4- LOOK_BACK      : int, gecmiş pencere (24-96 saat) - onceki paper araligindan
        5- HIDDEN         : int, BiLSTM gizli boyut (32-256)
        6- LAYERS         : int, LSTM katman sayisi (1-4)
        7- DROPOUT        : float, dropout orani (0.0-0.5)
        8- LR             : float (log), ogrenme orani (1e-4 - 1e-2)
        9- BATCH          : kategori, batch size (32, 64, 128)

    HORIZON = 1 sabit (kisa vade tahmin, onceki paper'da DWT+BiLSTM en iyi).
    """
    # -------- DWT (3) --------
    DWT_WAVELET = "dwt_wavelet"
    DWT_LEVEL   = "dwt_level"
    DWT_MODE    = "dwt_mode"

    # -------- DATA (1) --------
    LOOK_BACK = "look_back"

    # -------- BiLSTM (3) --------
    HIDDEN  = "hidden"
    LAYERS  = "layers"
    DROPOUT = "dropout"

    # -------- TRAINING (2) --------
    LR    = "lr"
    BATCH = "batch"


# =============================================================================
# 4. ARAMA UZAYI KATEGORILERI (param_mapping ve optimizers icin tek kaynak)
# =============================================================================

# Mother wavelet adaylari - ONCEKI PAPER (Supplementary Table S1) ile UYUMLU:
# Onceki paper'da TPE ile bulunan optimal: wavelet='sym5', level=6.
# Burada amac: 8 metasezgisel algoritmanin ayni arama uzayinda ayni veya daha iyi
# sonucu bulup bulamayacagini gostermek. Aralik tamamen onceki paper'la ayni.
WAVELET_CHOICES = [
    'db4', 'db5', 'db6', 'db7', 'db8',
    'sym5', 'sym6', 'sym7', 'sym8',
    'coif3', 'coif4', 'coif5',
]  # 12 secenek

# Onceki paper'daki PADP-tabanli (Optuna-TPE) optimal degerler (referans):
DWT_PRIOR_OPTIMAL = {'wavelet': 'sym5', 'level': 6}

# Padding modu - kenar etkilerini farkli sekilde isler
MODE_CHOICES = ['symmetric', 'zero', 'periodization']

# Batch size kategorileri
BATCH_CHOICES = [32, 64, 128]


# =============================================================================
# 5. OPTUNA TRIAL YARDIMCISI (referans amacli; metasezgisel arama da kullanilabilir)
# =============================================================================

def suggest_dwt_bilstm_params(trial):
    """Optuna trial uzerinde DWT+BiLSTM arama uzayini sunar.

    Args:
        trial: optuna.trial.Trial

    Returns:
        dict[HP, value]: 9 hiperparametreli dict
    """
    return {
        HP.DWT_WAVELET: trial.suggest_categorical(HP.DWT_WAVELET.value, WAVELET_CHOICES),  # 1
        HP.DWT_LEVEL:   trial.suggest_int(HP.DWT_LEVEL.value, 3, 7),                       # 2
        HP.DWT_MODE:    trial.suggest_categorical(HP.DWT_MODE.value, MODE_CHOICES),        # 3

        HP.LOOK_BACK:   trial.suggest_int(HP.LOOK_BACK.value, 24, 96),                     # 4

        HP.HIDDEN:      trial.suggest_int(HP.HIDDEN.value, 32, 256),                       # 5
        HP.LAYERS:      trial.suggest_int(HP.LAYERS.value, 1, 4),                          # 6
        HP.DROPOUT:     trial.suggest_float(HP.DROPOUT.value, 0.0, 0.5),                   # 7

        HP.LR:          trial.suggest_float(HP.LR.value, 1e-4, 1e-2, log=True),            # 8
        HP.BATCH:       trial.suggest_categorical(HP.BATCH.value, BATCH_CHOICES),          # 9
    }


# =============================================================================
# 6. DATASET YARDIMCILARI (mimari-agnostic, eskiden korundu)
# =============================================================================

def create_dataset(data, look_back):
    """Sliding window dataset olusturucu.

    Args:
        data: 1D numpy array, normalize edilmis tek bileSen sinyali
        look_back: int, geçmiş pencere uzunlugu

    Returns:
        X: shape (N - look_back - 1, look_back, 1)
        y: shape (N - look_back - 1, 1)
    """
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


def normalize_train_test_split(component, split_index):
    """IMF / DWT bileSenini train/test'e boler ve normalize eder.

    KRITIK: Scaler sadece train verisine fit edilir; data leakage onlenir.

    Args:
        component: 1D numpy array, tek IMF/sub-band sinyali
        split_index: int, train/test ayrim noktasi

    Returns:
        train_normalized: shape (split_index, 1)
        test_normalized:  shape (N - split_index, 1)
        scaler: sklearn.preprocessing.MinMaxScaler (inverse_transform icin)
    """
    component = component.reshape(-1, 1)

    train = component[:split_index]
    test  = component[split_index:]

    scaler = MinMaxScaler()
    train_normalized = scaler.fit_transform(train)
    test_normalized  = scaler.transform(test)

    return train_normalized, test_normalized, scaler


# =============================================================================
# 7. DWT YARDIMCILARI (yeni - PyWavelets sarmalayicilar)
# =============================================================================

def dwt_decompose(signal, wavelet='db4', level=3, mode='symmetric'):
    """Tek sinyali DWT ile ayriStir.

    Args:
        signal: 1D numpy array, ham sinyal
        wavelet: str, mother wavelet (haar/db4/db8/sym4/coif2)
        level: int, ayriStirma seviyesi (1-5)
        mode: str, padding ('symmetric'/'zero'/'periodization')

    Returns:
        coeffs: list of arrays, [cA_n, cD_n, ..., cD_1]
                cA_n: approximation (en kaba)
                cD_i: detail i'inci seviye

    Notlar:
        - Bileşen sayisi = level + 1 (1 approximation + level detail)
        - len(signal) uzunsa, level otomatik gecerli max'a tutturulur
    """
    import pywt
    # Maksimum gecerli level (sinyalin uzunluguna ve wavelet'e gore)
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    level_eff = min(int(level), max_level)
    coeffs = pywt.wavedec(signal, wavelet=wavelet, mode=mode, level=level_eff)
    return coeffs


def dwt_reconstruct(coeffs, wavelet='db4', mode='symmetric'):
    """DWT katsayilarindan sinyali geri olustur (inverse DWT).

    Args:
        coeffs: list, dwt_decompose ciktisi (veya tahmin edilmiş bileSenler)
        wavelet: str, ayriStirmada kullanilan wavelet (ayni olmali)
        mode: str, padding modu

    Returns:
        signal: 1D numpy array, yeniden olusturulmus sinyal
    """
    import pywt
    return pywt.waverec(coeffs, wavelet=wavelet, mode=mode)
