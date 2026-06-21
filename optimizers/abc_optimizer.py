# -*- coding: utf-8 -*-
"""
optimizers/abc_optimizer.py — Artificial Bee Colony Optimizer.

Karaboga (2005) tarafından önerilen ABC algoritmasının paralel uygulaması.
BaseOptimizer'dan türer; sadece ABC-spesifik fazlar (employed, onlooker,
scout) ve trial counter mantığı override edilir.

Referans: Karaboga, D. (2005). An idea based on honey bee swarm for
numerical optimization (Technical report TR06). Erciyes University.

Algoritma-spesifik parametreler:
    limit: Bir kaynağın terk edilmeden önceki başarısız trial sayısı.
           Klasik öneri: SN × D (food_number × dimension), pratik için
           genelde 5-10 arası ayarlanır. Bizim makaleler için sensitivity
           analizi yapıldı: 3, 5, 7 değerleri test edildi.
"""

import time
import random
import numpy as np
from typing import Dict, Any, List, Tuple

from optimizers import register_algorithm
from optimizers.base_optimizer import BaseOptimizer


@register_algorithm("abc")
class ABCOptimizer(BaseOptimizer):
    """Artificial Bee Colony Optimizer.

    Üç fazlı klasik ABC:
    1. Employed Bees: Her arı kendi kaynağına komşu arar
    2. Onlooker Bees: Fitness orantılı seçim ile kaynaklara gider
    3. Scout Bees: Limit aşan kaynakları rastgele yenile
    """

    DEFAULT_PARAMS = {
        'limit': 5,  # Sensitivity analizinde 3, 5, 7 test edildi
    }

    def _setup_algorithm_specific(self) -> None:
        """ABC-spesifik attribute'lar.

        Klasik ABC: food_number = pop_size / 2 (employed = onlooker = food_number).
        Pratik basitlik için pop_size'ı doğrudan food_number kabul ediyoruz;
        her cycle'da employed + onlooker = 2 × food_number evaluation, plus
        scout. Bu, GA'nın pop_size × max_iter budget'ı ile eşdeğer kalır.
        """
        self.food_number = self.pop_size  // 2
        self.limit = self.algo_params.get('limit', 5)

    # =========================================================================
    # POPÜLASYON BAŞLATMA
    # =========================================================================

    def _initialize_population(self) -> None:
        """Random foods, sıfır fitness, sonsuz f, sıfır trial."""
        self.population = np.random.rand(self.food_number, self.dim)
        self.f = np.full(self.food_number, float('inf'))
        self.fitness = np.zeros(self.food_number)
        self.trial = np.zeros(self.food_number, dtype=int)

    # =========================================================================
    # STATE SERIALIZATION (ABC-spesifik trial counter eklenir)
    # =========================================================================

    def _get_state_dict(self) -> Dict[str, Any]:
        d = super()._get_state_dict()
        d['trial'] = self.trial
        return d

    def _set_state_dict(self, state: Dict[str, Any]) -> None:
        super()._set_state_dict(state)
        # food_number property olduğu için resume sırasında otomatik doğru değer
        self.trial = state.get('trial', np.zeros(self.food_number, dtype=int))

    # =========================================================================
    # ABC OPERATÖRLERİ
    # =========================================================================

    def _abc_mutation(self, idx: int) -> np.ndarray:
        """ABC standart komşuluk arama operatörü.

        v_ij = x_ij + φ_ij × (x_ij - x_kj)
        burada k ≠ i rastgele seçilir, j rastgele bir boyut, φ ∈ [-1, 1].
        """
        neighbors = [i for i in range(self.food_number) if i != idx]
        neighbor = random.choice(neighbors)
        param = random.randint(0, self.dim - 1)

        new_sol = np.copy(self.population[idx])
        phi = random.uniform(-1.0, 1.0)
        new_sol[param] += phi * (
            self.population[idx][param] - self.population[neighbor][param]
        )
        return np.clip(new_sol, 0.0, 1.0)

    # =========================================================================
    # ANA ITERASYON: 3 FAZ
    # =========================================================================

    def _run_iteration(self, iteration: int) -> None:
        """Tek bir cycle: employed → onlooker → scout."""
        cycle_start = time.time()
        all_results: List[Dict] = []

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"CYCLE {iteration+1}/{self.max_iterations}")

        # 1. EMPLOYED PHASE
        emp_results = self._employed_phase(iteration)
        all_results.extend(emp_results)

        # 2. ONLOOKER PHASE
        onl_results = self._onlooker_phase(iteration)
        all_results.extend(onl_results)

        # 3. SCOUT PHASE
        sct_results, scouts_triggered = self._scout_phase(iteration)
        all_results.extend(sct_results)

        # Cycle istatistikleri
        duration = time.time() - cycle_start
        self.save_iteration_stats(
            iteration, all_results, duration,
            extra={'scouts_triggered': scouts_triggered}
        )
        self.save_checkpoint(iteration, 'CYCLE_COMPLETE')

        self.logger.info(
            f"\n[CYCLE {iteration+1}] Best: {self.global_opt:.6f} | "
            f"Duration: {duration:.1f}s | Scouts: {scouts_triggered}"
        )

    def _employed_phase(self, iteration: int) -> List[Dict]:
        """Employed bees: her kaynak için bir komşu üret ve değerlendir."""
        self.logger.info("[EMPLOYED] Çalışıyor...")
        candidates = [self._abc_mutation(i) for i in range(self.food_number)]
        results = self.evaluate(
            vectors=candidates,
            indices=list(range(self.food_number)),
            member_type="E",
            iteration=iteration,
        )

        improvements = []
        for i, result in enumerate(results):
            is_better = result['fitness'] > self.fitness[i]
            improvements.append(is_better)
            if is_better:
                self.population[i] = candidates[i]
                self.fitness[i] = result['fitness']
                self.f[i] = result['loss']
                self.trial[i] = 0
                self.update_global_best(i, iteration, 'EMPLOYED', result)
            else:
                self.trial[i] += 1

        self.save_trials(iteration, 'EMPLOYED', results, improvements)
        self.save_checkpoint(iteration, 'EMPLOYED')

        improved_count = sum(improvements)
        self.logger.info(
            f"[EMPLOYED] {improved_count}/{self.food_number} iyileşti"
        )
        return results

    def _onlooker_phase(self, iteration: int) -> List[Dict]:
        """Onlooker bees: fitness orantılı seçim ile kaynaklara git."""
        self.logger.info("[ONLOOKER] Çalışıyor...")

        # Olasılık hesabı
        max_fit = max(np.max(self.fitness), 1e-10)
        probs = 0.9 * (self.fitness / max_fit) + 0.1

        # Roulette wheel benzeri seçim (food_number kez)
        candidates = []
        indices = []
        t, i = 0, 0
        # Sonsuz döngü koruması: en fazla 10 × food_number deneme
        max_attempts = 10 * self.food_number
        attempts = 0
        while t < self.food_number and attempts < max_attempts:
            if random.random() < probs[i]:
                candidates.append(self._abc_mutation(i))
                indices.append(i)
                t += 1
            i = (i + 1) % self.food_number
            attempts += 1

        results = self.evaluate(
            vectors=candidates,
            indices=indices,
            member_type="O",
            iteration=iteration,
        )

        improvements = []
        for k, idx in enumerate(indices):
            result = results[k]
            is_better = result['fitness'] > self.fitness[idx]
            improvements.append(is_better)
            if is_better:
                self.population[idx] = candidates[k]
                self.fitness[idx] = result['fitness']
                self.f[idx] = result['loss']
                self.trial[idx] = 0
                self.update_global_best(idx, iteration, 'ONLOOKER', result)
            else:
                self.trial[idx] += 1

        self.save_trials(iteration, 'ONLOOKER', results, improvements)
        self.save_checkpoint(iteration, 'ONLOOKER')

        improved_count = sum(improvements)
        self.logger.info(
            f"[ONLOOKER] {improved_count}/{len(results)} iyileşti"
        )
        return results

    def _scout_phase(self, iteration: int) -> Tuple[List[Dict], int]:
        """Scout bees: limit aşan kaynakları rastgele yenile."""
        scout_indices = np.where(self.trial >= self.limit)[0]
        scouts_triggered = len(scout_indices)

        if scouts_triggered == 0:
            return [], 0

        self.logger.info(
            f"[SCOUT] {scouts_triggered} kaynak terk ediliyor..."
        )

        new_foods = [np.random.rand(self.dim) for _ in scout_indices]
        for i, idx in enumerate(scout_indices):
            self.population[idx] = new_foods[i]
            self.trial[idx] = 0

        results = self.evaluate(
            vectors=new_foods,
            indices=list(scout_indices),
            member_type="S",
            iteration=iteration,
        )

        improvements = []
        for k, idx in enumerate(scout_indices):
            result = results[k]
            self.f[idx] = result['loss']
            self.fitness[idx] = result['fitness']
            improved = self.update_global_best(idx, iteration, 'SCOUT', result)
            improvements.append(improved)

        self.save_trials(iteration, 'SCOUT', results, improvements)
        return results, scouts_triggered
