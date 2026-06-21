# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 18:04:31 2026

@author: EMRE5
"""

import torch
import os

# ---------------------------------------------------------
# HİPERPARAMETRELER VE AYARLAR
# ---------------------------------------------------------
LOOK_BACK = 24
TRAIN_RATIO = 0.8

DEVICE_OPTUNA = torch.device("cpu")   # Optuna CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STORAGE = "sqlite:///optuna_wind.db"
STUDY_NAME = "vmd_lstm_wind"

# ---------------------------------------------------------
# DOSYA VE KLASÖR YOLLARI
# ---------------------------------------------------------
# Projenin ana dizini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Klasör İsimleri
DATA_DIR_NAME = "data_files"
MODELS_DIR_NAME = "models"     
FIGURES_DIR_NAME = "figures"   
# Tam Yollar
DATA_DIR_PATH = os.path.join(BASE_DIR, DATA_DIR_NAME)
MODELS_DIR_PATH = os.path.join(BASE_DIR, MODELS_DIR_NAME)     
FIGURES_DIR_PATH = os.path.join(BASE_DIR, FIGURES_DIR_NAME)   

# Klasörler yoksa oluştur (Garanti olsun)
os.makedirs(DATA_DIR_PATH, exist_ok=True)
os.makedirs(MODELS_DIR_PATH, exist_ok=True)
os.makedirs(FIGURES_DIR_PATH, exist_ok=True)

# Dosya Yolları
DATA_FILE_NAME = "data.xlsx"
DATA_FILE_PATH = os.path.join(DATA_DIR_PATH, DATA_FILE_NAME)


All_STATION_DATA_NAME = "all_station_data.npy"
All_STATION_DATA_PATH = os.path.join(DATA_DIR_PATH, All_STATION_DATA_NAME)

def get_station_data_path(station_name):
    # safe_name = station_name.replace(" ", "_").replace("/", "_")
    return os.path.join(DATA_DIR_PATH, f"{station_name}_data.npy")


# ---------------------------------------------------------
# DİĞER SABİTLER
# ---------------------------------------------------------
# WARMP_UP_RATE ve PENALTY_WEIGHT v9'da kaldirildi (PADP attildi).
OPT_DATA_LIMIT = 8760  # 365*24 = bir yillik veri - Optimizasyon veri limiti