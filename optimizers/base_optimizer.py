# -*- coding: utf-8 -*-
"""
optimizers/base_optimizer.py — Soyut metasezgisel optimizer iskeleti.

Tüm metasezgisel algoritmalar bu sınıftan türer. Ortak işler (paralel
fitness evaluation, checkpoint/resume, DB yönetimi, seed kontrolü, equal
computational budget enforcement, global best takibi) burada tek seferlik
yazılmıştır.

Yeni algoritma eklerken yapılacaklar:
1. BaseOptimizer'dan türeyen yeni sınıf yaz
2. @register_algorithm("name") decorator'ı ile kaydet
3. _initialize_population() ve _run_iteration() abstract metodlarını override et
4. (Opsiyonel) _get_state_dict() / _set_state_dict() ile algoritma-spesifik
   state'i checkpoint'e ekle (hız, momentum, trial counters, vb.)

Detay için optimizers/README.md.
"""

import os
import time
import random
import logging
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from optimizers.db_manager import GenericDBManager
from optimizers.param_mapping import DIMENSION

# evaluator modülü torch + vmdpy yükler; lazy import içerden yapılır
# (smoke test ve test ortamı bağımlılıkları azalır)


class BaseOptimizer(ABC):
    """Soyut metasezgisel optimizer.

    Alt sınıflar şunları sağlamalı:
        ALGORITHM_NAME: str — registry için (decorator ile set edilir)
        DEFAULT_PARAMS: dict — algoritma-spesifik varsayılan parametreler

    Soyut metodlar:
        _initialize_population()
        _run_iteration(iteration)

    Override edilebilir metodlar:
        _get_state_dict(), _set_state_dict()
    """

    # Alt sınıf override etmeli (decorator ile otomatik set edilir)
    ALGORITHM_NAME: str = "base"

    # Alt sınıf override edebilir
    DEFAULT_PARAMS: Dict[str, Any] = {}

    # =========================================================================
    # KURULUM
    # =========================================================================

    def __init__(self,
                 db_file: str,
                 seed: Optional[int] = None,
                 runid: Optional[int] = None,
                 pop_size: int = 40,
                 max_iterations: int = 50,
                 dimension: int = DIMENSION,
                 n_workers: Optional[int] = None,
                 algo_params: Optional[Dict[str, Any]] = None,
                 mode: str = "joint",
                 archive_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            db_file: SQLite veritabanı dosyası
            seed: Random seed (None ise time.time() kullanılır)
            runid: Paralel çalışma ID'si (logging için)
            pop_size: Popülasyon büyüklüğü
            max_iterations: Maksimum cycle/generation/iter
            dimension: Vektör boyutu (genelde 10)
            n_workers: Paralel worker sayısı (None: cpu_count()-1)
            algo_params: Algoritma-spesifik parametreler (varsayılanlar
                         DEFAULT_PARAMS'tan alınır, override edilirse update)
            mode: "joint" veya "sequential" — sequential mode için alt sınıf
                  ek mantık uygulayabilir
            archive_dir: Tamamlanmış DB'lerin arşivleneceği klasör
            logger: Custom logger (None: default)
        """
        # Temel parametreler
        self.db_file = db_file
        self.runid = runid
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.dim = dimension
        self.mode = mode

        # Worker sayısı
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)
        self.n_workers = n_workers

        # Algoritma-spesifik parametreler
        self.algo_params = dict(self.DEFAULT_PARAMS)
        if algo_params:
            self.algo_params.update(algo_params)

        # Seed yönetimi (deterministic davranış için)
        if seed is None:
            seed = int(time.time())
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Logger
        self.logger = logger or logging.getLogger(f"OPT.{self.ALGORITHM_NAME.upper()}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

        # Arşiv dizini
        self.archive_dir = archive_dir or f"{self.ALGORITHM_NAME}_archive"

        # Durum değişkenleri (alt sınıf doldurur)
        self.population: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None
        self.f: Optional[np.ndarray] = None  # Loss değerleri
        self.global_opt: float = float('inf')
        self.global_vector: np.ndarray = np.zeros(self.dim)

        # Devam noktası (checkpoint'ten)
        self.start_iteration: int = 0
        self.start_phase: str = 'INIT'

        # DB
        db_dir = os.path.dirname(self.db_file)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self.db = GenericDBManager(self.db_file, algorithm=self.ALGORITHM_NAME)

        # Algoritma-spesifik attribute'ları başlat (subclass override eder).
        # Bu hook _initialize_db_state'ten ÖNCE çağrılır ki populasyon
        # başlatma sırasında ihtiyaç duyulan attribute'lar hazır olsun.
        self._setup_algorithm_specific()

        # DB'yi başlat veya checkpoint'ten devam et
        self._initialize_db_state()

    def _setup_algorithm_specific(self) -> None:
        """Algoritma-spesifik attribute kurulum hook'u.

        Bu metod _initialize_db_state'ten ÖNCE çağrılır. Burada subclass'lar
        kendi attribute'larını set edebilir (örn. ABC için food_number ve limit;
        PSO için velocity sınırları; vb.).

        Default impl: hiçbir şey yapmaz. Override opsiyoneldir.
        """
        pass

    def _initialize_db_state(self) -> None:
        """DB durumunu inceler: yeni başlat / devam et / arşivle+yeni başlat."""
        status = self.db.get_status()

        if status == 'COMPLETED':
            self._archive_and_restart()
        elif status == 'RUNNING':
            self.logger.info("Yarım kalan çalışma tespit edildi.")
            db_algo = self.db.get_algorithm()
            if db_algo and db_algo != self.ALGORITHM_NAME:
                self.logger.error(
                    f"DB '{db_algo}' algoritması ile başlatılmış, "
                    f"şu an '{self.ALGORITHM_NAME}' çalıştırılıyor. "
                    f"Farklı DB dosyası kullan veya eskisini sil."
                )
                raise RuntimeError("Algorithm mismatch in DB")
            if not self._resume_from_checkpoint():
                self.logger.warning("Checkpoint yüklenemedi, sıfırdan başlanıyor.")
                self._start_new_run()
        else:
            self._start_new_run()

    def _start_new_run(self) -> None:
        """Yeni bir çalışma başlatır."""
        self.db.initialize_run(
            seed=self.seed,
            pop_size=self.pop_size,
            max_iter=self.max_iterations,
            dimension=self.dim,
            config={
                'algo_params': self.algo_params,
                'mode': self.mode,
                'runid': self.runid,
            }
        )
        # Alt sınıf populasyonu başlatır
        self._initialize_population()
        self.start_iteration = 0
        self.start_phase = 'INIT'
        self.logger.info(
            f"Yeni {self.ALGORITHM_NAME.upper()} çalışması başlatıldı | "
            f"Seed: {self.seed} | Pop: {self.pop_size} | "
            f"MaxIter: {self.max_iterations}"
        )

    def _archive_and_restart(self) -> None:
        """Tamamlanmış eski DB'yi arşivler ve yeni çalışma başlatır."""
        os.makedirs(self.archive_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(
            self.archive_dir, f"{self.ALGORITHM_NAME}_{ts}.db"
        )
        try:
            shutil.move(self.db_file, archive_path)
            self.logger.info(f"Önceki çalışma arşivlendi: {archive_path}")
        except Exception as e:
            self.logger.warning(f"Arşivleme hatası: {e}")
        # DB taşındı, yeni manager oluştur
        self.db = GenericDBManager(self.db_file, algorithm=self.ALGORITHM_NAME)
        self._start_new_run()

    def _resume_from_checkpoint(self) -> bool:
        """Checkpoint'ten devam eder. Başarılıysa True."""
        cp = self.db.load_checkpoint()
        if cp is None:
            return False

        self.start_iteration = cp['iteration']
        self.start_phase = cp['phase']
        self.global_opt = cp['global_opt']
        self.global_vector = cp['global_vector']

        # Alt sınıf, state dict'ten algoritma-spesifik state'i yükler
        self._set_state_dict(cp['state'])

        # Cycle tamamlanmışsa bir sonrakine geç
        if self.start_phase in ('CYCLE_COMPLETE', 'ITER_COMPLETE',
                                 'GENERATION_COMPLETE'):
            self.start_iteration += 1
            self.start_phase = 'INIT_NEXT'

        self.logger.info(
            f"Checkpoint yüklendi | Iter: {self.start_iteration} | "
            f"Phase: {self.start_phase} | Best: {self.global_opt:.6f}"
        )
        return True

    # =========================================================================
    # ÇEKİRDEK API (alt sınıf çağırır)
    # =========================================================================

    def evaluate(self,
                 vectors: List[np.ndarray],
                 indices: List[int],
                 member_type: str,
                 iteration: int) -> List[Dict]:
        """Verilen vektörleri paralel değerlendirir.

        Args:
            vectors: Değerlendirilecek [0,1]^D vektörleri (list of arrays)
            indices: Her vektörün popülasyondaki ID'si
            member_type: Trial tipi etiketi ("I", "E", "O", "S", "P", "C", vb.)
            iteration: Mevcut iterasyon numarası

        Returns:
            evaluator.evaluate_solution_wrapper çıktıları listesi
        """
        if not vectors:
            return []

        # Lazy import: torch/vmdpy yüklenir
        from optimizers.evaluator import evaluate_solution_wrapper, init_worker

        tasks = [
            (np.asarray(v), i, member_type, iteration, self.runid, self.mode)
            for v, i in zip(vectors, indices)
        ]

        # Tek-worker durumunda Pool gereksiz overhead olur
        if self.n_workers <= 1:
            return [evaluate_solution_wrapper(t) for t in tasks]

        with Pool(processes=self.n_workers, initializer=init_worker) as pool:
            results = pool.map(evaluate_solution_wrapper, tasks)
        return results

    def update_global_best(self, idx: int, iteration: int,
                           phase: str, result: Dict) -> bool:
        """Bir bireyin sonucu varsa global best'i günceller.

        Args:
            idx: Bireyin popülasyondaki ID'si
            iteration: Iter no
            phase: Faz adı
            result: evaluate_solution_wrapper çıktısı

        Returns:
            True if global best güncellendi, False değişmedi
        """
        loss = result['loss']
        if loss >= self.global_opt:
            return False

        old_best = self.global_opt
        self.global_opt = loss
        self.global_vector = np.copy(self.population[idx]) \
            if self.population is not None else np.asarray(result['vector'])

        self.logger.info(f"  >>> YENİ GLOBAL BEST: {self.global_opt:.6f}")

        self.db.save_best_improvement(
            iteration=iteration,
            phase=phase,
            global_opt=self.global_opt,
            params_str=result['params_str'],
            vector=self.global_vector,
            old_best=old_best,
        )
        return True

    def save_checkpoint(self, iteration: int, phase: str) -> None:
        """Mevcut durumu DB'ye kaydeder."""
        state = self._get_state_dict()
        self.db.save_checkpoint(
            iteration=iteration,
            phase=phase,
            state=state,
            global_opt=self.global_opt,
            global_vector=self.global_vector,
        )

    def save_trials(self, iteration: int, phase: str,
                    results: List[Dict],
                    improvements: Optional[List[bool]] = None) -> None:
        """Toplu trial kaydı."""
        self.db.save_trials_batch(iteration, phase, results, improvements)

    def save_iteration_stats(self, iteration: int, results: List[Dict],
                              duration: float, extra: Optional[Dict] = None) -> None:
        """Iterasyon istatistiklerini kaydeder."""
        valid = [r['loss'] for r in results if r['loss'] < float('inf')]
        stats = {
            'eval_count': len(results),
            'best_loss': min(valid) if valid else None,
            'worst_loss': max(valid) if valid else None,
            'mean_loss': float(np.mean(valid)) if valid else None,
            'std_loss': float(np.std(valid)) if valid else None,
            'global_best': self.global_opt,
            'duration': duration,
            'extra': extra or {},
        }
        self.db.save_iteration_stats(iteration, stats)

    # =========================================================================
    # STATE YÖNETİMİ (alt sınıf override edebilir)
    # =========================================================================

    def _get_state_dict(self) -> Dict[str, Any]:
        """Checkpoint'e yazılacak state dict.

        Default impl: population, fitness, f. Alt sınıf ek alanlar için
        bu metodu çağırıp kendi alanlarını ekler:

            def _get_state_dict(self):
                d = super()._get_state_dict()
                d['velocities'] = self.velocities  # PSO için
                return d
        """
        return {
            'population': self.population,
            'fitness': self.fitness,
            'f': self.f,
        }

    def _set_state_dict(self, state: Dict[str, Any]) -> None:
        """Checkpoint'ten state'i yükler. Alt sınıf override edebilir."""
        self.population = state.get('population')
        self.fitness = state.get('fitness')
        self.f = state.get('f')

    # =========================================================================
    # SOYUT METODLAR (alt sınıf zorunlu impl)
    # =========================================================================

    @abstractmethod
    def _initialize_population(self) -> None:
        """Popülasyonu rastgele başlatır.

        Alt sınıf en azından şunu yapmalı:
            self.population = np.random.rand(self.pop_size, self.dim)
            self.fitness = np.zeros(self.pop_size)
            self.f = np.full(self.pop_size, float('inf'))

        Algoritma-spesifik state (hız, trial counter, vb.) burada init edilir.
        """
        ...

    @abstractmethod
    def _run_iteration(self, iteration: int) -> None:
        """Tek bir iterasyon (cycle/generation) çalıştırır.

        Bu metod içinde alt sınıf:
        1. Yeni adayları üretir (mutation/crossover/velocity update/vb.)
        2. self.evaluate() ile paralel değerlendirir
        3. Popülasyonu günceller (replacement, selection, vb.)
        4. Her improvement için self.update_global_best() çağırır
        5. self.save_trials() ile DB'ye yazar
        6. self.save_checkpoint() ile state kaydeder
        7. self.save_iteration_stats() ile cycle istatistikleri yazar
        """
        ...

    # =========================================================================
    # ANA ÇALIŞMA DÖNGÜSÜ
    # =========================================================================

    def run(self) -> Tuple[np.ndarray, float]:
        """Optimizasyonu çalıştırır.

        Returns:
            (best_vector, best_loss) tuple
        """
        self.logger.info("=" * 60)
        self.logger.info(
            f"{self.ALGORITHM_NAME.upper()} OPTIMIZASYONU"
        )
        self.logger.info("=" * 60)
        self.logger.info(
            f"Pop: {self.pop_size} | MaxIter: {self.max_iterations} | "
            f"Workers: {self.n_workers} | Seed: {self.seed} | Mode: {self.mode}"
        )
        self.logger.info(f"Algo params: {self.algo_params}")

        start_total = time.time()

        # INIT phase: ilk popülasyonu değerlendir (sadece hiç başlamamışsa)
        if self.start_iteration == 0 and self.start_phase == 'INIT':
            self._run_init_phase()

        # Ana iterasyon döngüsü
        for it in range(self.start_iteration, self.max_iterations):
            self._run_iteration(it)

        # Bitir
        self.db.mark_completed()
        total_time = (time.time() - start_total) / 60.0

        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"TAMAMLANDI | Süre: {total_time:.1f} dakika")
        self.logger.info(f"Best loss: {self.global_opt:.6f}")
        self.logger.info("=" * 60)

        return self.global_vector, self.global_opt

    def _run_init_phase(self) -> None:
        """Başlangıç popülasyonunu değerlendirir.

        Not: ABC gibi `food_number = pop_size // 2` kullanan algoritmalar icin
        population matrix boyutu pop_size'tan farkli olabilir; bu yuzden
        pop_size yerine population.shape[0] kullaniyoruz (generic).
        """
        self.logger.info("\n[INIT] Popülasyon değerlendiriliyor...")

        n_init = int(self.population.shape[0])  # ABC icin food_number, digerleri icin pop_size
        results = self.evaluate(
            vectors=[self.population[i] for i in range(n_init)],
            indices=list(range(n_init)),
            member_type="I",
            iteration=0,
        )

        for i, result in enumerate(results):
            if self.f is not None:
                self.f[i] = result['loss']
            if self.fitness is not None:
                self.fitness[i] = result['fitness']
            self.update_global_best(i, 0, 'INIT', result)

        self.save_trials(0, 'INIT', results)
        self.save_checkpoint(0, 'INIT_COMPLETE')
        self.start_phase = 'ITER_START'

        self.logger.info(f"[INIT] Best: {self.global_opt:.6f}")
