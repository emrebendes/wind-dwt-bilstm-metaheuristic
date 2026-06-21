# -*- coding: utf-8 -*-
"""
objective.py - DWT + BiLSTM joint optimization fitness fonksiyonu (v8).

Mimari:
    - Sinyal ayri Stirma: pywt.wavedec (DWT)
    - Model: BiLSTMModel (model.py), her bileSen icin ayri BiLSTM
    - Forecast horizon: H=1 (onceki paper'da DWT+BiLSTM en iyi)
    - Final tahmin: Inverse DWT (pywt.waverec) ile bileSenleri birleStir

Iki kullanim modu:
    1) Optimizasyon Modu (is_final_training=False):
       - CPU (paralel worker'lar GPU paylasamaz)
       - 1-yil veri segmenti (OPT_DATA_LIMIT)
       - Rastgele 1 istasyon (mode=global) veya sabit istasyon (mode=single)
       - 20 epoch + early stopping patience=5

    2) Final EGitim Modu (is_final_training=True):
       - GPU (CUDA varsa)
       - Tum 8 istasyon birlesik (mode=global)
       - Tam veri (6 yil)
       - 100 epoch + early stopping patience=15
       - Model + grafik kaydeder

Ablation flagleri (Faz 4 deneyleri icin):
    use_dwt           : False -> DWT atlanir, ham sinyal tek "bilesen" gibi (level=1)
    use_bidirectional : False -> BiLSTM yerine duz LSTM

Return contract:
    Default       : float (toplam loss = bileSen ortalama RMSE)
    return_components=True (opsiyonel) : (total_loss, component_rmses)
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # headless backend, multiprocessing-safe
import matplotlib.pyplot as plt

import pywt

from config import (
    TRAIN_RATIO,
    DEVICE_OPTUNA,
    DEVICE,
    MODELS_DIR_PATH,
    FIGURES_DIR_PATH,
    OPT_DATA_LIMIT,
)
from model import BiLSTMModel, EarlyStopping
from utils import (
    HP,
    WAVELET_CHOICES,
    MODE_CHOICES,
    create_dataset,
    setup_logger,
    normalize_train_test_split,
    load_ruzgar_verisi,
    dwt_decompose,
    dwt_reconstruct,
)


# =============================================================================
# ANA SINIF
# =============================================================================

class WindForecastObjective:
    """DWT + BiLSTM joint optimization icin polymorphic fitness siniifi.

    Kullanim modlari:
      1) Optuna / metasezgisel (is_final_training=False) - CPU, hizli, 1 istasyon
      2) Final egitim (is_final_training=True) - GPU, yavas, tum istasyon
    """

    def __init__(self,
                 mode: str = "global",
                 station_idx: int = 0,
                 is_final_training: bool = False,
                 show: int = 0,
                 use_gpu=None,
                 use_dwt: bool = True,
                 use_bidirectional: bool = True,
                 data_window: tuple = None):
        """
        Args:
            mode: "global" (rastgele/tum istasyon) | "single" (sabit istasyon)
            station_idx: "single" modda hangi istasyon
            is_final_training: True ise GPU + tam veri + 100 epoch
            show: log siklikgi (0=kapali, N=her N epoch'ta)
            use_gpu: None (otomatik) | True (zorla GPU) | False (zorla CPU)

        Ablation flagleri (Faz 4):
            use_dwt           : False -> DWT kapali (ham sinyal, level_effective=0)
            use_bidirectional : False -> duz LSTM (BiLSTMModel(bidirectional=False))
        """
        self.mode = mode
        self.station_idx = station_idx
        self.is_final = is_final_training
        self.show = show

        # Ablation bayrakları
        self.use_dwt = use_dwt
        self.use_bidirectional = use_bidirectional

        # Walk-forward: ozel veri penceresi (Faz 2)
        # None ise default davranis: HP search son OPT_DATA_LIMIT saati alir.
        # (start, end) tuple verilirse: signals[start:end] alinir.
        # Final training (is_final=True) BU PARAMETREYI YOK SAYAR — tam veri kullanir.
        self.data_window = data_window

        # Veriyi yukle
        self.data_list = load_ruzgar_verisi()

        # Logger
        self.logger = setup_logger(
            "FINAL_ENGINE" if is_final_training else "OPT_WORKER"
        )

        # Cihaz secimi
        if use_gpu is True:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif use_gpu is False:
            self.device = torch.device("cpu")
        else:
            # Otomatik: optimizasyon CPU, final GPU varsa GPU
            self.device = DEVICE if (is_final_training and torch.cuda.is_available()) \
                          else DEVICE_OPTUNA

    # =========================================================================
    # YARDIMCI: ISTASYON SECIMI
    # =========================================================================

    def _get_signals_to_process(self):
        """Modu gore islenecek istasyon sinyallerini doner."""
        # Final + Global: TUM ISTASYONLAR
        if self.is_final and self.mode == "global":
            self.logger.info(
                f"Final/Global mode: {len(self.data_list)} istasyonun "
                f"tamami isleniyor."
            )
            return list(self.data_list)

        # Optimization + Global: RASTGELE 1 ISTASYON (hiz icin)
        if self.mode == "global":
            idx = random.randint(0, len(self.data_list) - 1)
            return [self.data_list[idx]]

        # Single mode: SABIT ISTASYON
        return [self.data_list[self.station_idx]]

    # =========================================================================
    # YARDIMCI: TEK BILESEN BiLSTM EGITIMI
    # =========================================================================

    def _train_one_component(self, model, loader, X_val, y_val,
                             lr, epochs, patience, tag, comp_idx=0):
        """Tek DWT bileseni icin BiLSTM egitir, en iyi val RMSE'yi doner.

        Args:
            model: BiLSTMModel ornegi (zaten device'da)
            loader: DataLoader (train)
            X_val, y_val: tensor (validation, device'da)
            lr, epochs, patience: training hiperparametreleri
            tag: log prefix string
            comp_idx: bilesen indeksi (sadece log icin)

        Returns:
            best_rmse: float (early-stopping ile elde edilen en iyi val RMSE)
            train_losses, val_losses: list[float] (her epoch MSE'leri)
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        es = EarlyStopping(patience=patience)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for Xb, yb in loader:
                optimizer.zero_grad()
                pred = model(Xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())
            avg_train_loss = float(np.mean(batch_losses))

            # Validation (mini-batch ile -- GPU OOM oncesi)
            model.eval()
            with torch.no_grad():
                val_batch = max(int(loader.batch_size), 32)
                n_val = X_val.size(0)
                total_loss, count = 0.0, 0
                for s in range(0, n_val, val_batch):
                    e = min(s + val_batch, n_val)
                    vp = model(X_val[s:e])
                    bl = loss_fn(vp, y_val[s:e]).item()
                    total_loss += bl * (e - s)
                    count += (e - s)
                val_loss = total_loss / max(count, 1)

            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)

            if self.show > 0 and epoch % self.show == 0:
                self.logger.info(
                    f"{tag} | comp {comp_idx+1} | epoch {epoch:3d} | "
                    f"train={avg_train_loss:.6f} | val={val_loss:.6f}"
                )

            es(val_loss, model)
            if es.stop:
                if self.is_final:
                    self.logger.info(
                        f"{tag} | comp {comp_idx+1} | early-stop @ epoch {epoch}"
                    )
                break

        es.restore(model)

        # ---------------------------------------------------------------------
        # En iyi val RMSE (early-stopping ile elde edilen)
        # PADP v9'da kaldirildi (Faz 3 ablation gosterdi: nötr, p=0.71).
        # ---------------------------------------------------------------------
        best_rmse = float(np.sqrt(es.best_loss))

        # Final modda her bileSen icin egitim grafigi kaydet
        if self.is_final:
            try:
                os.makedirs(FIGURES_DIR_PATH, exist_ok=True)
                plt.figure(figsize=(10, 5))
                plt.plot(train_losses, label='Train Loss')
                plt.plot(val_losses, label='Val Loss', linestyle='--')
                plt.title(f'Global Model - Bilesen {comp_idx+1} Loss')
                plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
                fig_path = os.path.join(
                    FIGURES_DIR_PATH, f'global_comp{comp_idx+1}_loss.png'
                )
                plt.savefig(fig_path); plt.close()
                self.logger.info(
                    f"--> Bilesen {comp_idx+1} kaybi grafigi: {fig_path}"
                )

                # Model checkpoint
                os.makedirs(MODELS_DIR_PATH, exist_ok=True)
                ckpt = os.path.join(
                    MODELS_DIR_PATH, f'global_model_comp{comp_idx+1}.pth'
                )
                torch.save(model.state_dict(), ckpt)
                self.logger.info(f"--> Bilesen {comp_idx+1} modeli: {ckpt}")
            except Exception as e:
                self.logger.warning(f"Grafik/model kaydedilemedi: {e}")

        return best_rmse, train_losses, val_losses

    # =========================================================================
    # ANA CAGRI
    # =========================================================================

    def __call__(self, params: dict, log_prefix: str = "",
                 return_components: bool = False):
        """params dict alir, total loss doner.

        Args:
            params: dict[HP, value] - 9 hiperparametre (param_mapping ciktisi)
            log_prefix: str - log satirlari icin tag (run/iter/birey)
            return_components: True ise (total_loss, component_rmses) tuple doner;
                               PADP diagnostic'in bilesen-arasi varyans hesabi icin

        Returns:
            float (penalized loss) veya (float, list[float]) tuple
        """
        # 1) Parametreleri cek
        wavelet = params[HP.DWT_WAVELET]
        level   = int(params[HP.DWT_LEVEL])
        mode    = params[HP.DWT_MODE]
        look_back = int(params[HP.LOOK_BACK])
        hidden   = int(params[HP.HIDDEN])
        layers   = int(params[HP.LAYERS])
        dropout  = float(params[HP.DROPOUT])
        lr       = float(params[HP.LR])
        batch    = int(params[HP.BATCH])

        self.logger.info(
            f"{log_prefix} "
            f"wavelet={wavelet} | level={level} | mode={mode} | "
            f"look_back={look_back} | hidden={hidden} | layers={layers} | "
            f"dropout={dropout:.3f} | lr={lr:.5f} | batch={batch} | "
            f"ablation[dwt={self.use_dwt}, bi={self.use_bidirectional}]"
        )

        # 2) Sinyalleri al
        signals = self._get_signals_to_process()

        # Optimizasyonda sadece son 1 yili al (hiz)
        # Walk-forward: self.data_window verilirse o aralik kullanilir.
        processed_signals = []
        if not self.is_final:
            if self.data_window is not None:
                # Faz 2 walk-forward: ozel pencere
                w_start, w_end = self.data_window
                for s in signals:
                    s = np.array(s, dtype=np.float64)
                    seg = s[w_start:w_end]
                    if len(seg) < 200:  # minimum sinyal uzunlugu sanity
                        self.logger.warning(
                            f"Pencere [{w_start}:{w_end}] cok kisa "
                            f"(len={len(seg)}), atlandi."
                        )
                        continue
                    processed_signals.append(seg)
            else:
                # Default: son OPT_DATA_LIMIT saat
                for s in signals:
                    s = np.array(s, dtype=np.float64)
                    if len(s) > OPT_DATA_LIMIT:
                        processed_signals.append(s[-OPT_DATA_LIMIT:])
                    else:
                        processed_signals.append(s)
        else:
            # Final training her zaman tam veri kullanir.
            processed_signals = [np.array(s, dtype=np.float64) for s in signals]

        # 3) DWT ayriStirma
        # Ablation use_dwt=False ise: ham sinyal tek "bilesen" olarak ele alinir
        # (bunu degerlendirme amaciyla level_effective=0 kullaniyoruz, sonuc:
        # K_eff=1 bilesen).
        if not self.use_dwt:
            # NoDWT: her istasyon icin tek bilesen = ham sinyal
            decomposed = [[s.copy()] for s in processed_signals]
            n_components = 1
        else:
            decomposed = []
            for s in processed_signals:
                try:
                    coeffs = dwt_decompose(
                        s, wavelet=wavelet, level=level, mode=mode
                    )  # list of arrays: [cA_n, cD_n, ..., cD_1]
                    decomposed.append(coeffs)
                except Exception as e:
                    self.logger.error(f"DWT hatasi: {e}")
                    return (float('inf'), []) if return_components else float('inf')
            # En kucuk bilesen sayisini al (level otomatik tutturulmus olabilir)
            n_components = min(len(c) for c in decomposed)

        # 4) Her bilesen icin train/test havuzu hazirla (cross-istasyon birlesik)
        component_pools = [
            {'X_tr': [], 'y_tr': [], 'X_te': [], 'y_te': []}
            for _ in range(n_components)
        ]
        valid_sig_count = 0

        for coeffs in decomposed:
            for i in range(n_components):
                comp_signal = np.asarray(coeffs[i], dtype=np.float64).flatten()
                if len(comp_signal) < look_back + 10:
                    continue
                split_idx = int(len(comp_signal) * TRAIN_RATIO)
                tr_n, te_n, _ = normalize_train_test_split(comp_signal, split_idx)
                X_tr, y_tr = create_dataset(tr_n, look_back)
                X_te, y_te = create_dataset(te_n, look_back)
                if len(X_tr) > 0 and len(X_te) > 0:
                    component_pools[i]['X_tr'].append(X_tr)
                    component_pools[i]['y_tr'].append(y_tr)
                    component_pools[i]['X_te'].append(X_te)
                    component_pools[i]['y_te'].append(y_te)
            valid_sig_count += 1

        if valid_sig_count == 0:
            self.logger.warning("Hic gecerli sinyal yok, inf donduruluyor.")
            return (float('inf'), []) if return_components else float('inf')

        # 5) Her bilesen icin BiLSTM egit + skor topla
        component_rmses = []   # ham val RMSE (early-stopping ile)

        epochs   = 100 if self.is_final else 20
        patience = 15  if self.is_final else 5

        for i in range(n_components):
            pool = component_pools[i]
            if not pool['X_tr']:
                continue

            X_tr = np.concatenate(pool['X_tr'], axis=0)
            y_tr = np.concatenate(pool['y_tr'], axis=0)
            X_te = np.concatenate(pool['X_te'], axis=0)
            y_te = np.concatenate(pool['y_te'], axis=0)

            # Tensor + device
            Xtr_t = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
            ytr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1).to(self.device)
            Xte_t = torch.tensor(X_te, dtype=torch.float32).to(self.device)
            yte_t = torch.tensor(y_te, dtype=torch.float32).view(-1, 1).to(self.device)

            loader = DataLoader(
                TensorDataset(Xtr_t, ytr_t),
                batch_size=batch, shuffle=True
            )

            model = BiLSTMModel(
                input_size=1,
                hidden_size=hidden,
                num_layers=layers,
                dropout=dropout,
                bidirectional=self.use_bidirectional,
            ).to(self.device)

            best_rmse, _, _ = self._train_one_component(
                model, loader, Xte_t, yte_t,
                lr=lr, epochs=epochs, patience=patience,
                tag=log_prefix, comp_idx=i,
            )
            component_rmses.append(best_rmse)

        if not component_rmses:
            return (float('inf'), []) if return_components else float('inf')

        # 6) Toplam skor = bilesen RMSE'lerinin ortalamasi
        total_score = float(np.mean(component_rmses))

        if self.is_final:
            self.logger.info(
                f"{log_prefix} FINAL toplam skor: {total_score:.6f} | "
                f"bilesen RMSE'leri: "
                f"[{', '.join(f'{r:.4f}' for r in component_rmses)}]"
            )

        if return_components:
            return total_score, component_rmses
        return total_score


# =============================================================================
# SMOKE TEST (cli)
# =============================================================================

if __name__ == "__main__":
    print("=== objective.py smoke test ===")
    obj = WindForecastObjective(mode="global", is_final_training=False, show=0)
    from utils import HP
    params = {
        HP.DWT_WAVELET: 'sym5',
        HP.DWT_LEVEL: 4,
        HP.DWT_MODE: 'symmetric',
        HP.LOOK_BACK: 48,
        HP.HIDDEN: 64,
        HP.LAYERS: 2,
        HP.DROPOUT: 0.2,
        HP.LR: 1e-3,
        HP.BATCH: 64,
    }
    loss = obj(params, log_prefix="[SMOKE]")
    print(f"Smoke test loss: {loss:.4f}")
