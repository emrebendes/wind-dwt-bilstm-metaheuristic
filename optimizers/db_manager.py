# -*- coding: utf-8 -*-
"""
optimizers/db_manager.py — Generic veritabanı yöneticisi.

Tüm metasezgisel algoritmalar için ortak SQLite tabanlı kalıcı durum
yönetimi. WAL mode + batch insert + algoritma-bağımsız checkpoint state.

Özellikler:
- WAL mode: Yazma sırasında okuma bloklanmaz
- Batch insert: Çoklu trial kaydı tek transaction'da
- Checkpoint blob: Algoritma-spesifik state pickle ile saklanır
  → Yeni algoritma eklerken DB schema değiştirmeye gerek yok
- TRUBA 3-gün limit'ine dayanıklı: her iterasyon sonu commit
"""

import os
import sqlite3
import pickle
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

import numpy as np


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar/array types.

    PSO ve GA gibi algoritmalarda `improvements` listesi numpy.bool_
    icerir (numpy array ile karsilastirma sonucu), bu durum
    sum(improvements) -> numpy.int64 yapar ve standart json.dumps fail eder.
    Bu encoder tum numpy tiplerini Python yerlilerine cevirir.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class GenericDBManager:
    """Generic optimizer DB yöneticisi.

    Tek bir DB dosyası tek bir optimizasyon çalışmasını içerir.
    Algoritma-spesifik state `checkpoint.state_blob` BLOB'unda saklanır.
    """

    def __init__(self, db_file: str, algorithm: str = "unknown"):
        """
        Args:
            db_file: SQLite veritabanı dosya yolu
            algorithm: Algoritma adı (metadata için), örn. 'abc', 'ga', 'pso'
        """
        self.db_file = db_file
        self.algorithm = algorithm
        self._lock = threading.Lock()  # Thread-safe yazma için

    @contextmanager
    def _connection(self):
        """Context manager ile bağlantı yönetimi."""
        conn = sqlite3.connect(self.db_file, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        try:
            yield conn
        finally:
            conn.close()

    # =========================================================================
    # ŞEMA OLUŞTURMA
    # =========================================================================

    def create_tables(self) -> None:
        """Tüm tabloları oluşturur (idempotent)."""
        with self._connection() as conn:
            cur = conn.cursor()

            # Metadata: tek satır, optimizasyon meta-bilgileri
            cur.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY,
                    algorithm TEXT,
                    status TEXT,
                    started_at TEXT,
                    last_updated_at TEXT,
                    completed_at TEXT,
                    random_seed INTEGER,
                    population_size INTEGER,
                    max_iterations INTEGER,
                    dimension INTEGER,
                    total_evaluations INTEGER DEFAULT 0,
                    config_json TEXT
                )
            """)

            # Checkpoint: algoritma-spesifik state pickle BLOB olarak
            cur.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint (
                    id INTEGER PRIMARY KEY,
                    iteration INTEGER,
                    phase TEXT,
                    state_blob BLOB,
                    global_opt REAL,
                    global_vector BLOB,
                    updated_at TEXT
                )
            """)

            # Trial geçmişi: her bir fitness evaluation'ın kaydı
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trial_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration INTEGER,
                    phase TEXT,
                    member_id INTEGER,
                    member_type TEXT,
                    loss REAL,
                    fitness REAL,
                    dwt_wavelet TEXT,
                    dwt_level INTEGER,
                    dwt_mode TEXT,
                    look_back INTEGER,
                    hidden INTEGER,
                    layers INTEGER,
                    dropout REAL,
                    lr REAL,
                    batch INTEGER,
                    vector BLOB,
                    elapsed_seconds REAL,
                    is_improvement INTEGER DEFAULT 0,
                    created_at TEXT
                )
            """)

            # Best geçmişi: her global best iyileşmesi
            cur.execute("""
                CREATE TABLE IF NOT EXISTS best_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration INTEGER,
                    phase TEXT,
                    global_opt REAL,
                    dwt_wavelet TEXT,
                    dwt_level INTEGER,
                    dwt_mode TEXT,
                    look_back INTEGER,
                    hidden INTEGER,
                    layers INTEGER,
                    dropout REAL,
                    lr REAL,
                    batch INTEGER,
                    vector BLOB,
                    improvement_from REAL,
                    improvement_percent REAL,
                    created_at TEXT
                )
            """)

            # Iteration istatistikleri
            cur.execute("""
                CREATE TABLE IF NOT EXISTS iteration_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration INTEGER,
                    eval_count INTEGER,
                    best_loss REAL,
                    worst_loss REAL,
                    mean_loss REAL,
                    std_loss REAL,
                    global_best_at_end REAL,
                    duration_seconds REAL,
                    extra_json TEXT,
                    created_at TEXT
                )
            """)

            # İndeksler
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trial_iter ON trial_history(iteration)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trial_loss ON trial_history(loss)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_best_iter ON best_history(iteration)")

            conn.commit()

    # =========================================================================
    # METADATA & DURUM
    # =========================================================================

    def initialize_run(self, seed: int, pop_size: int, max_iter: int,
                       dimension: int, config: Optional[Dict] = None) -> None:
        """Yeni çalışma başlatır (eski DB'yi siler).

        Args:
            seed: Random seed
            pop_size: Popülasyon büyüklüğü
            max_iter: Maksimum iterasyon (cycle/generation)
            dimension: Vektör boyutu (genelde 10)
            config: Algoritma-spesifik config (örn: ABC limit, GA crossover_rate)
        """
        if os.path.exists(self.db_file):
            os.remove(self.db_file)

        self.create_tables()

        config_json = json.dumps(config or {}, cls=_NumpyJSONEncoder)

        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO metadata
                (id, algorithm, status, started_at, random_seed,
                 population_size, max_iterations, dimension, config_json)
                VALUES (1, ?, 'RUNNING', ?, ?, ?, ?, ?, ?)
            """, (
                self.algorithm,
                datetime.now().isoformat(),
                seed, pop_size, max_iter, dimension, config_json
            ))
            conn.commit()

    def get_status(self) -> Optional[str]:
        """Çalışma durumunu döndürür ('RUNNING', 'COMPLETED', None)."""
        if not os.path.exists(self.db_file):
            return None
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT status FROM metadata WHERE id=1")
            row = cur.fetchone()
            return row[0] if row else None

    def get_algorithm(self) -> Optional[str]:
        """DB'deki algoritma adını döndürür."""
        if not os.path.exists(self.db_file):
            return None
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT algorithm FROM metadata WHERE id=1")
            row = cur.fetchone()
            return row[0] if row else None

    def mark_completed(self) -> None:
        """Çalışmayı 'COMPLETED' işaretler."""
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE metadata SET status='COMPLETED', completed_at=? WHERE id=1",
                (datetime.now().isoformat(),)
            )
            conn.commit()

    def update_evaluation_count(self, increment: int) -> None:
        """Toplam evaluation sayacını günceller."""
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE metadata SET total_evaluations = total_evaluations + ?, "
                "last_updated_at = ? WHERE id=1",
                (increment, datetime.now().isoformat())
            )
            conn.commit()

    # =========================================================================
    # CHECKPOINT
    # =========================================================================

    def save_checkpoint(self, iteration: int, phase: str,
                        state: Dict[str, Any],
                        global_opt: float,
                        global_vector: np.ndarray) -> None:
        """Checkpoint kaydeder.

        Args:
            iteration: Mevcut cycle/generation numarası
            phase: Faz adı (ABC: 'EMPLOYED', 'ONLOOKER', 'SCOUT', 'CYCLE_COMPLETE'.
                   GA: 'GENERATION_COMPLETE'. PSO: 'ITER_COMPLETE'. vb.)
            state: Algoritma-spesifik state dict (population, fitness, trial counters,
                   velocities, vb.) — pickle ile serialize edilir.
            global_opt: Mevcut global en iyi loss değeri
            global_vector: Global en iyi vektör
        """
        with self._lock:
            with self._connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO checkpoint
                    (id, iteration, phase, state_blob, global_opt, global_vector, updated_at)
                    VALUES (1, ?, ?, ?, ?, ?, ?)
                """, (
                    iteration, phase,
                    pickle.dumps(state),
                    float(global_opt),
                    pickle.dumps(np.asarray(global_vector)),
                    datetime.now().isoformat()
                ))
                cur.execute(
                    "UPDATE metadata SET last_updated_at=? WHERE id=1",
                    (datetime.now().isoformat(),)
                )
                conn.commit()

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Checkpoint yükler. Bulunamazsa None döner."""
        if not os.path.exists(self.db_file):
            return None
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT iteration, phase, state_blob, global_opt, global_vector
                FROM checkpoint WHERE id=1
            """)
            row = cur.fetchone()
            if row is None:
                return None
            return {
                'iteration': row[0],
                'phase': row[1],
                'state': pickle.loads(row[2]),
                'global_opt': row[3],
                'global_vector': pickle.loads(row[4]),
            }

    # =========================================================================
    # TRIAL & BEST KAYIT
    # =========================================================================

    def save_trials_batch(self, iteration: int, phase: str,
                          results: List[Dict],
                          improvements: Optional[List[bool]] = None) -> None:
        """Toplu trial kaydı (tek transaction'da, hızlı).

        Args:
            iteration: Iterasyon numarası
            phase: Faz adı
            results: evaluator.evaluate_solution_wrapper çıktıları
            improvements: Her trial'ın iyileşme olup olmadığı (opsiyonel)
        """
        if not results:
            return

        if improvements is None:
            improvements = [False] * len(results)

        rows = []
        for r, imp in zip(results, improvements):
            ps = r['params_str']
            rows.append((
                iteration, phase,
                r['member_id'], r['member_type'],
                r['loss'], r['fitness'],
                ps.get('dwt_wavelet'), ps.get('dwt_level'), ps.get('dwt_mode'),
                ps.get('look_back'),
                ps.get('hidden'), ps.get('layers'),
                ps.get('dropout'), ps.get('lr'), ps.get('batch'),
                pickle.dumps(np.asarray(r['vector'])),
                r['elapsed_seconds'],
                1 if imp else 0,
                r['timestamp'],
            ))

        with self._lock:
            with self._connection() as conn:
                cur = conn.cursor()
                cur.executemany("""
                    INSERT INTO trial_history
                    (iteration, phase, member_id, member_type, loss, fitness,
                     dwt_wavelet, dwt_level, dwt_mode,
                     look_back, hidden, layers,
                     dropout, lr, batch, vector, elapsed_seconds,
                     is_improvement, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                cur.execute(
                    "UPDATE metadata SET total_evaluations = total_evaluations + ? "
                    "WHERE id=1",
                    (len(results),)
                )
                conn.commit()

    def save_best_improvement(self, iteration: int, phase: str,
                               global_opt: float,
                               params_str: Dict,
                               vector: np.ndarray,
                               old_best: float) -> None:
        """Global best iyileşme kaydı."""
        if old_best == float('inf'):
            improvement_pct = 0.0
        else:
            improvement_pct = ((old_best - global_opt) / old_best * 100.0)

        with self._lock:
            with self._connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO best_history
                    (iteration, phase, global_opt,
                     dwt_wavelet, dwt_level, dwt_mode,
                     look_back, hidden, layers,
                     dropout, lr, batch, vector,
                     improvement_from, improvement_percent, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    iteration, phase, global_opt,
                    params_str.get('dwt_wavelet'), params_str.get('dwt_level'),
                    params_str.get('dwt_mode'),
                    params_str.get('look_back'), params_str.get('hidden'),
                    params_str.get('layers'),
                    params_str.get('dropout'),
                    params_str.get('lr'), params_str.get('batch'),
                    pickle.dumps(np.asarray(vector)),
                    old_best, improvement_pct,
                    datetime.now().isoformat()
                ))
                conn.commit()

    def save_iteration_stats(self, iteration: int, stats: Dict) -> None:
        """Iterasyon istatistiklerini kaydeder.

        Args:
            stats: dict, anahtarlar:
                eval_count, best_loss, worst_loss, mean_loss, std_loss,
                global_best, duration, extra (algoritma-spesifik dict)
        """
        extra_json = json.dumps(stats.get('extra', {}), cls=_NumpyJSONEncoder)
        with self._lock:
            with self._connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO iteration_stats
                    (iteration, eval_count, best_loss, worst_loss,
                     mean_loss, std_loss, global_best_at_end,
                     duration_seconds, extra_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    iteration,
                    stats.get('eval_count', 0),
                    stats.get('best_loss'),
                    stats.get('worst_loss'),
                    stats.get('mean_loss'),
                    stats.get('std_loss'),
                    stats.get('global_best'),
                    stats.get('duration'),
                    extra_json,
                    datetime.now().isoformat()
                ))
                conn.commit()

    # =========================================================================
    # ANALİZ / SORGULAMA
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Çalışmanın özetini döndürür."""
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM metadata WHERE id=1")
            meta_row = cur.fetchone()

            cur.execute("SELECT COUNT(*) FROM trial_history")
            total = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM trial_history WHERE is_improvement=1")
            improvements = cur.fetchone()[0]

            cur.execute("SELECT MIN(loss) FROM trial_history WHERE loss < 999999")
            best = cur.fetchone()[0]

            return {
                'metadata': meta_row,
                'total_trials': total,
                'improvements': improvements,
                'best_loss': best,
            }

    def get_best_params(self) -> Optional[Dict]:
        """En iyi parametreleri döndürür."""
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT dwt_wavelet, dwt_level, dwt_mode,
                       look_back, hidden, layers,
                       dropout, lr, batch, loss
                FROM trial_history WHERE loss < 999999
                ORDER BY loss ASC LIMIT 1
            """)
            row = cur.fetchone()
            if row is None:
                return None
            return {
                'dwt_wavelet': row[0], 'dwt_level': row[1], 'dwt_mode': row[2],
                'look_back': row[3], 'hidden': row[4], 'layers': row[5],
                'dropout': row[6], 'lr': row[7], 'batch': row[8],
                'loss': row[9],
            }
        """En iyi parametreleri döndürür."""
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT dwt_wavelet, dwt_level, dwt_mode,
                       look_back, hidden, layers,
                       dropout, lr, batch, loss
                FROM trial_history WHERE loss < 999999
                ORDER BY loss ASC LIMIT 1
            """)
            row = cur.fetchone()
            if row is None:
                return None
            return {
                'dwt_wavelet': row[0], 'dwt_level': row[1], 'dwt_mode': row[2],
                'look_back': row[3], 'hidden': row[4], 'layers': row[5],
                'dropout': row[6], 'lr': row[7], 'batch': row[8],
                'loss': row[9],
            }
