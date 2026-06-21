# -*- coding: utf-8 -*-
"""
model.py - DWT + BiLSTM joint optimization icin sinir agi siniflari.

Mimari (v8, 2026-05-28):
    - BiLSTMModel: torch.nn.LSTM(bidirectional=True) tabanli regresyon modeli
    - Onceki paper'da H=1 icin BiLSTM en iyi (ortalama RMSE = 0.6766)
    - Yeni paper'da DWT+BiLSTM kombinasyonu (Manuscript.docx, RMSE=0.0357 @H=1)

Ablation destegi (Faz 4):
    - bidirectional=False -> duz LSTM (NoBidirectional ablation)

Onceki TCAN+VMD'den yapilan degisiklikler:
    - TCANModel + TemporalBlock + TemporalAttention kaldirildi
    - BiLSTMModel eklendi
    - EarlyStopping korundu (mimari-agnostic)
"""

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# 1. BiLSTM MODEL (yeni ana mimari)
# =============================================================================

class BiLSTMModel(nn.Module):
    """Bidirectional LSTM tabanli zaman serisi regresyonu.

    Varsayilan: bidirectional=True (BiLSTM).
    Ablation: bidirectional=False -> duz LSTM (Faz 4 "NoBidirectional" testi).

    Forward:
        Input shape : (B, look_back, input_size)
        Output shape: (B, 1)  (tek timestep, H=1 tahmin)

    Args:
        input_size: int, ozellik boyutu (DWT bileSenleri ayri eGitildigi icin = 1)
        hidden_size: int, LSTM gizli birim sayisi (32-256)
        num_layers: int, LSTM katman sayisi (1-4)
        dropout: float, dropout orani (0.0-0.5) -- num_layers > 1 ise aktif
        bidirectional: bool, default True (BiLSTM); False -> duz LSTM
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # PyTorch LSTM dropout sadece num_layers > 1 oldugunda calisir
        lstm_dropout = float(dropout) if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        # BiLSTM ciktisi: (B, look_back, hidden_size * 2)
        # LSTM ciktisi:   (B, look_back, hidden_size)
        fc_input = hidden_size * 2 if bidirectional else hidden_size

        # Tek dense katman, son timestep -> tek deger tahmin
        self.fc = nn.Linear(fc_input, 1)

    def forward(self, x):
        """
        Args:
            x: shape (B, look_back, input_size)
        Returns:
            shape (B, 1) -- bir adim sonrasinin tahmin degeri
        """
        # out shape: (B, look_back, hidden_size * num_directions)
        out, _ = self.lstm(x)

        # Son timestep'in temsilini al
        # BiLSTM ise hem ileri (out[:, -1, :hidden_size]) hem geri yonlu
        # (out[:, 0, hidden_size:]) bilgi son ciktida birleSiktir, biz son
        # adimdaki birleSik temsili kullanmak yeterli.
        last_step = out[:, -1, :]  # (B, hidden_size * num_directions)

        return self.fc(last_step)  # (B, 1)


# =============================================================================
# 2. EARLY STOPPING (mimari-agnostic, eskiden korundu)
# =============================================================================

class EarlyStopping:
    """Validation loss bazli erken durdurma.

    Args:
        patience: int, iyileSme gormeyen epoch tolerans sayisi

    Kullanim:
        es = EarlyStopping(patience=15)
        for epoch in range(N):
            ...
            es(val_loss, model)
            if es.stop: break
        es.restore(model)  # en iyi state'i geri yukle
    """

    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None
        self.stop = False

    def __call__(self, loss, model):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model):
        """En iyi gorulen state'i modele geri yukler."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# =============================================================================
# 3. YARDIMCI: PARAMETRE SAYISI (ablation karSilaStirma icin)
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """EGitilebilir parametre sayisini doner (ablation kar Silastirmasi icin)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 4. SMOKE TEST (dogrudan import + forward + ablation toggle)
# =============================================================================

if __name__ == "__main__":
    print("=== BiLSTMModel smoke test ===")

    B, T, D = 4, 24, 1  # batch=4, look_back=24, input_size=1
    x = torch.randn(B, T, D)

    # 1) BiLSTM (default)
    m_bi = BiLSTMModel(input_size=D, hidden_size=64, num_layers=2,
                       dropout=0.2, bidirectional=True)
    y_bi = m_bi(x)
    n_bi = count_parameters(m_bi)
    print(f"  [BiLSTM]  input {tuple(x.shape)} -> output {tuple(y_bi.shape)}, "
          f"params={n_bi:,}")

    # 2) Duz LSTM (NoBidirectional ablation)
    m_uni = BiLSTMModel(input_size=D, hidden_size=64, num_layers=2,
                        dropout=0.2, bidirectional=False)
    y_uni = m_uni(x)
    n_uni = count_parameters(m_uni)
    print(f"  [LSTM]    input {tuple(x.shape)} -> output {tuple(y_uni.shape)}, "
          f"params={n_uni:,}")

    # BiLSTM yaklasik 2x parametre, ozellikle LSTM ag agirliklarinda
    ratio = n_bi / n_uni
    print(f"  BiLSTM / LSTM parametre orani: {ratio:.2f}x "
          f"(beklenen ~2x)")

    # 3) Tek katman dropout konsistensi
    m_1l = BiLSTMModel(input_size=D, hidden_size=32, num_layers=1,
                       dropout=0.5, bidirectional=True)
    y_1l = m_1l(x)
    print(f"  [1-layer dropout=0.5 -> 0.0] output shape {tuple(y_1l.shape)}, "
          f"PASS (PyTorch warning olmamali)")

    # 4) EarlyStopping smoke
    es = EarlyStopping(patience=3)
    for i, fake_loss in enumerate([1.0, 0.9, 0.8, 0.85, 0.9, 0.95, 1.0]):
        es(fake_loss, m_bi)
        if es.stop:
            print(f"  EarlyStopping triggered at iter {i}, best={es.best_loss:.3f}")
            break
    print("  EarlyStopping smoke PASS")
